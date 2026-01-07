"""
S3 Upload node for LangGraph workflow
"""

import base64
import logging
from functools import lru_cache
from typing import Any, Optional

import boto3
import fitz  # PyMuPDF
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from ...config import get_processing_config, get_upload_config
from ...models.schemas import WorkflowStatus
from ..state import WorkflowState

logger = logging.getLogger(__name__)

# Boto3 config with connection pooling and reasonable timeouts
_boto_config = BotoConfig(
    connect_timeout=5,
    read_timeout=30,
    max_pool_connections=10,
    retries={"max_attempts": 3, "mode": "standard"},
)


def _update_workflow_progress(
    group_id: str,
    identifier_name: str,
    step_id: str,
    step_name: str,
    status: WorkflowStatus,
    message: str,
) -> None:
    """
    Update workflow progress in the ContractService store.
    Uses late import to avoid circular dependency.
    """
    try:
        from ...services.contract_service import ContractService

        ContractService._update_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id=step_id,
            step_name=step_name,
            status=status,
            message=message,
        )
    except Exception as e:
        logger.warning(f"Failed to update workflow progress: {e}")


@lru_cache(maxsize=1)
def get_s3_client():
    """Get cached S3 client with configured region and connection pooling"""
    upload_config = get_upload_config()
    region = upload_config.get("region", "us-east-1")
    return boto3.client("s3", region_name=region, config=_boto_config)


def build_upload_path(identifier_name: str, file_type: str) -> str:
    """
    Build the S3 upload path for a document.
    Format: <path_prefix>/<file_type>/<identifier_name>/
    """
    upload_config = get_upload_config()
    path_prefix = upload_config.get("path_prefix", "uploaded")
    return f"{path_prefix}/{identifier_name}/{file_type}"


def _extract_pdf_pages(
    pdf_content: bytes,
    convert_to_image: bool = True,
    image_format: str = "jpeg",
    dpi: int = 150,
) -> tuple[list[bytes], str, str]:
    """
    Extract each page of a PDF, optionally converting to images using PyMuPDF (fitz).

    Args:
        pdf_content: Raw bytes of the PDF file
        convert_to_image: If True, convert pages to images; if False, keep as PDF
        image_format: Output image format (jpeg, png, etc.) when converting
        dpi: Resolution for the output images when converting

    Returns:
        Tuple of (list of page bytes, content_type, file_extension)
    """
    page_bytes_list = []

    # Open PDF from bytes
    pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

    try:
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            if convert_to_image:
                # Calculate zoom factor based on DPI (72 is the base DPI for PDF)
                zoom = dpi / 72
                matrix = fitz.Matrix(zoom, zoom)

                # Render page to pixmap
                pixmap = page.get_pixmap(matrix=matrix)

                # Convert to image bytes
                if image_format.lower() == "jpeg":
                    page_bytes = pixmap.tobytes("jpeg")
                elif image_format.lower() == "png":
                    page_bytes = pixmap.tobytes("png")
                else:
                    # Default to JPEG
                    page_bytes = pixmap.tobytes("jpeg")

                page_bytes_list.append(page_bytes)
            else:
                # Extract page as a single-page PDF
                single_page_pdf = fitz.open()
                single_page_pdf.insert_pdf(
                    pdf_document, from_page=page_num, to_page=page_num
                )
                page_bytes = single_page_pdf.tobytes()
                single_page_pdf.close()
                page_bytes_list.append(page_bytes)
    finally:
        pdf_document.close()

    # Determine content type and extension based on conversion mode
    if convert_to_image:
        content_type = f"image/{image_format.lower()}"
        file_extension = image_format.lower()
    else:
        content_type = "application/pdf"
        file_extension = "pdf"

    return page_bytes_list, content_type, file_extension


def _sync_upload_document(
    s3_client,
    bucket: str,
    s3_key: str,
    content: bytes,
    content_type: str,
    metadata: dict[str, str],
) -> None:
    """Synchronous S3 upload (to be run in thread pool)"""
    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=content,
        ContentType=content_type,
        Metadata=metadata,
    )


async def _upload_document_pages(
    s3_client,
    bucket: str,
    upload_path: str,
    doc_content: bytes,
    doc_filename: str,
    file_type: str,
    group_id: str,
    identifier_name: str,
    convert_to_image: bool = True,
    image_format: str = "jpeg",
    dpi: int = 150,
) -> tuple[list[str], dict[int, str]]:
    """
    Extract and upload each page of a document to S3.

    When convert_to_image is True, pages are converted to images (jpeg/png).
    When convert_to_image is False, pages are extracted as individual PDFs.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        upload_path: Base S3 path for uploads
        doc_content: Raw document bytes
        doc_filename: Original filename
        file_type: Document file type classification
        group_id: Group ID for metadata
        identifier_name: Identifier name for metadata
        convert_to_image: If True, convert pages to images; if False, keep as PDF
        image_format: Output image format (when converting to image)
        dpi: Output image DPI (when converting to image)

    Returns:
        Tuple of (list of uploaded S3 keys, dict mapping page numbers to S3 URIs)
    """
    import asyncio

    # Update progress: extracting pages
    _update_workflow_progress(
        group_id=group_id,
        identifier_name=identifier_name,
        step_id="extracting_pages",
        step_name="Document Upload",
        status=WorkflowStatus.IN_PROGRESS,
        message=f"Extracting pages from {doc_filename}...",
    )

    # Extract pages (either as images or as individual PDFs)
    page_bytes_list, content_type, file_extension = _extract_pdf_pages(
        doc_content, convert_to_image, image_format, dpi
    )

    total_pages = len(page_bytes_list)
    uploaded_keys = []
    page_uri_mapping: dict[int, str] = {}

    for page_num, page_bytes in enumerate(page_bytes_list, start=1):
        # Build S3 key: <upload_path>/<base_filename>/<page_number>.<extension>
        s3_key = f"{upload_path}/{page_num}.{file_extension}"

        # Upload to S3
        await asyncio.to_thread(
            _sync_upload_document,
            s3_client,
            bucket,
            s3_key,
            page_bytes,
            content_type,
            {
                "group_id": group_id,
                "identifier_name": identifier_name,
                "file_type": file_type,
                "original_filename": doc_filename,
                "page_number": str(page_num),
            },
        )

        uploaded_keys.append(s3_key)
        page_uri_mapping[page_num] = f"s3://{bucket}/{s3_key}"

        logger.info(
            f"[{group_id}] Successfully uploaded page {page_num} of {doc_filename}"
        )

    return uploaded_keys, page_uri_mapping


async def upload_documents(state: WorkflowState) -> dict[str, Any]:
    """
    Upload all documents in the document group to S3 (async).

    This node:
    1. Reads the document group from state
    2. Converts each document page to an image (if convert_to_image is enabled)
    3. Uploads each page image to S3 bucket at: uploaded/<identifier_name>/<file_type>/<filename>/<page_num>.<ext>
    4. Returns updated state with uploaded file keys and page image URIs

    Args:
        state: Current workflow state containing document_group

    Returns:
        Updated state fields with upload results
    """
    document_group = state["document_group"]
    group_id = document_group.group_id
    identifier_name = document_group.identifier_name

    logger.info(f"[{group_id}] Starting document upload to S3")

    # Update progress: starting upload
    _update_workflow_progress(
        group_id=group_id,
        identifier_name=identifier_name,
        step_id="upload_starting",
        step_name="Document Upload",
        status=WorkflowStatus.IN_PROGRESS,
        message=f"Starting upload for {len(document_group.documents)} document(s)...",
    )

    upload_config = get_upload_config()
    processing_config = get_processing_config()

    bucket = upload_config.get("bucket", "contract-assistant-workflow")

    # Get image conversion settings from config
    convert_to_image = processing_config.get("convert_to_image", True)
    image_format = processing_config.get("image_format", "jpeg")
    image_dpi = processing_config.get("image_dpi", 150)

    uploaded_files = []
    page_images: dict[str, dict[int, str]] = {}

    try:
        s3_client = get_s3_client()

        total_docs = len(document_group.documents)
        for doc_index, doc in enumerate(document_group.documents, start=1):
            # Update progress for each document
            _update_workflow_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id=f"processing_doc_{doc_index}",
                step_name="Document Upload",
                status=WorkflowStatus.IN_PROGRESS,
                message=f"Processing document {doc_index}/{total_docs}: {doc.filename}",
            )

            # Build S3 key: uploaded/<file_type>/<identifier_name>/<filename>
            upload_path = build_upload_path(identifier_name, doc.file_type.value)

            # Decode base64 content from document
            encoded_content = getattr(doc, "content", None)
            doc_content = base64.b64decode(encoded_content) if encoded_content else b""

            if doc_content:
                # Extract and upload each page (as images or PDFs based on config)
                keys, page_mapping = await _upload_document_pages(
                    s3_client=s3_client,
                    bucket=bucket,
                    upload_path=upload_path,
                    doc_content=doc_content,
                    doc_filename=doc.filename,
                    file_type=doc.file_type.value,
                    group_id=group_id,
                    identifier_name=identifier_name,
                    convert_to_image=convert_to_image,
                    image_format=image_format,
                    dpi=image_dpi,
                )
                uploaded_files.extend(keys)

                # Store page URIs under the file type
                if doc.file_type.value not in page_images:
                    page_images[doc.file_type.value] = {}
                page_images[doc.file_type.value].update(page_mapping)

        logger.info(
            f"[{group_id}] All {len(uploaded_files)} documents uploaded successfully"
        )

        # Update progress: upload complete
        _update_workflow_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="upload_complete",
            step_name="Document Upload",
            status=WorkflowStatus.IN_PROGRESS,
            message=f"Successfully uploaded {len(uploaded_files)} page(s) to S3",
        )

        path_prefix = upload_config.get("path_prefix", "uploaded")
        return {
            "status": WorkflowStatus.IN_PROGRESS,
            "current_step": "upload_complete",
            "uploaded_files": uploaded_files,
            "upload_path": f"s3://{bucket}/{path_prefix}",
            "page_images": page_images if page_images else None,
            "error_message": None,
        }

    except ClientError as e:
        error_msg = f"S3 upload failed: {str(e)}"
        logger.error(f"[{group_id}] {error_msg}")

        # Update progress: upload failed
        _update_workflow_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="upload_failed",
            step_name="Document Upload",
            status=WorkflowStatus.FAILED,
            message=error_msg,
        )

        return {
            "status": WorkflowStatus.FAILED,
            "current_step": "upload_failed",
            "uploaded_files": uploaded_files,
            "upload_path": None,
            "page_images": page_images if page_images else None,
            "error_message": error_msg,
        }

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        logger.error(f"[{group_id}] {error_msg}")

        # Update progress: upload failed
        _update_workflow_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="upload_failed",
            step_name="Document Upload",
            status=WorkflowStatus.FAILED,
            message=error_msg,
        )

        return {
            "status": WorkflowStatus.FAILED,
            "current_step": "upload_failed",
            "uploaded_files": uploaded_files,
            "upload_path": None,
            "page_images": page_images if page_images else None,
            "error_message": error_msg,
        }
