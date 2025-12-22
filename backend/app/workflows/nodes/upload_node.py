"""
S3 Upload node for LangGraph workflow
"""

import logging
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from ...config import get_upload_config
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


def _sync_upload_document(
    s3_client,
    bucket: str,
    s3_key: str,
    content_type: str,
    metadata: dict[str, str],
) -> None:
    """Synchronous S3 upload (to be run in thread pool)"""
    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=b"",  # Replace with actual file content
        ContentType=content_type,
        Metadata=metadata,
    )


async def upload_documents(state: WorkflowState) -> dict[str, Any]:
    """
    Upload all documents in the document group to S3 (async).

    This node:
    1. Reads the document group from state
    2. Uploads each document to S3 bucket at: uploaded/<identifier_name>/<filename>
    3. Returns updated state with uploaded file keys

    Args:
        state: Current workflow state containing document_group

    Returns:
        Updated state fields with upload results
    """
    import asyncio

    document_group = state["document_group"]
    group_id = document_group.group_id
    identifier_name = document_group.identifier_name

    logger.info(f"[{group_id}] Starting document upload to S3")

    upload_config = get_upload_config()
    bucket = upload_config.get("bucket", "contract-assistant-workflow")

    uploaded_files = []

    try:
        s3_client = get_s3_client()

        for doc in document_group.documents:
            # Build S3 key: uploaded/<file_type>/<identifier_name>/<filename>
            upload_path = build_upload_path(identifier_name, doc.file_type.value)
            s3_key = f"{upload_path}/{doc.filename}"

            logger.info(
                f"[{group_id}] Uploading {doc.filename} to s3://{bucket}/{s3_key}"
            )

            # Run synchronous S3 upload in thread pool to avoid blocking event loop
            await asyncio.to_thread(
                _sync_upload_document,
                s3_client,
                bucket,
                s3_key,
                doc.content_type or "application/octet-stream",
                {
                    "group_id": group_id,
                    "identifier_name": identifier_name,
                    "file_type": doc.file_type.value,
                    "original_filename": doc.filename,
                },
            )

            uploaded_files.append(s3_key)
            logger.info(f"[{group_id}] Successfully uploaded {doc.filename}")

        logger.info(
            f"[{group_id}] All {len(uploaded_files)} documents uploaded successfully"
        )

        path_prefix = upload_config.get("path_prefix", "uploaded")
        return {
            "status": WorkflowStatus.IN_PROGRESS,
            "current_step": "upload_complete",
            "uploaded_files": uploaded_files,
            "upload_path": f"s3://{bucket}/{path_prefix}",
            "error_message": None,
        }

    except ClientError as e:
        error_msg = f"S3 upload failed: {str(e)}"
        logger.error(f"[{group_id}] {error_msg}")
        return {
            "status": WorkflowStatus.FAILED,
            "current_step": "upload_failed",
            "uploaded_files": uploaded_files,
            "upload_path": None,
            "error_message": error_msg,
        }

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        logger.error(f"[{group_id}] {error_msg}")
        return {
            "status": WorkflowStatus.FAILED,
            "current_step": "upload_failed",
            "uploaded_files": uploaded_files,
            "upload_path": None,
            "error_message": error_msg,
        }
