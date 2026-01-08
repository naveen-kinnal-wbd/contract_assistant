"""
Base info extractor agent node for LangGraph workflow.

This node extracts basic contract metadata from document images using
AWS Bedrock LLM with parallel processing per page.
"""

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from config import get_llm_config, get_upload_config
from models.schemas import WorkflowStatus
from services.bedrock_client import BedrockClient
from workflows.state import WorkflowState
from workflows.nodes.base import BaseWorkflowNode

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPTS - Define system and user prompts for base info extraction
# ============================================================================

BASE_INFO_SCHEMA = {
    "schema": {
        "contract_type": {
            "description": "Specifies the type of agreement",
            "allowed_values": [
                "Acquisition",
                "Co-production",
                "Commission",
                "Amendment",
            ],
            "additional_instructions": [],
            "output_format": "string",
        },
        "contract_name": {
            "description": "Text describing the contract",
            "allowed_values": "ANY",
            "additional_instructions": [
                "If the contract type is Acquisition: it should be License Agreement - <Program>"
            ],
            "output_format": "string",
        },
        "date_executed": {
            "description": "Date when the contract is signed",
            "allowed_values": "ANY",
            "additional_instructions": [
                "Look in: 1) the signature section 2) the header/footer of document 3) at the beginning of the document in the format of 'As of <Date>' in that order",
                "If you could not find any, leave it blank",
                "In case of multiple documents, if there is an 'Attachment' document type, extract from Attachment; otherwise extract from Master document",
            ],
            "output_format": "string",
        },
        "date_effective": {
            "description": "Date when the contract goes into effect",
            "allowed_values": "ANY",
            "additional_instructions": [
                "Look for phrases like 'Dated as <Date>' or 'as of <Date>'",
                "In case of multiple documents, if there is an 'Attachment' document type, extract from Attachment; otherwise extract from Master document",
            ],
            "output_format": "string",
        },
        "parties": {
            "description": "List of companies and individuals named in the contract; should always include WBD or one of its subsidiaries",
            "allowed_values": "ANY",
            "additional_instructions": [
                "Should have Value of party and Type",
                "A WBD entity will always be of type 'Primary'; all others will be 'Contracting entity'",
                "You should have one 'Primary' and one 'Contracting entity'",
                "Warner Bros. Domestic is not a WBD subsidiary",
            ],
            "output_format": "list",
        },
        "programs": {
            "description": "List of titles of a movie, series, or other media asset being acquired or produced for exhibition.",
            "allowed_values": "ANY",
            "additional_instructions": [],
            "output_format": "list",
        },
    }
}

SYSTEM_PROMPT = f"""
    You are an expert contract analyst and document understanding system.

    Your task is to extract metadata from a SINGLE PAGE of a media contract.
    The page is provided as an image. 
    Take the provided JSON Schema <INPUT_SCHEMA> as a reference to understand the fields, its description, its output format, and its allowed values.

    <INPUT_SCHEMA>
    {BASE_INFO_SCHEMA}
    </INPUT_SCHEMA>

    CRITICAL RULES:
    - When extracting metadata, you must follow the list of additional_instructions for each JSON field that is specified in the <INPUT_SCHEMA> schema, if it exists. The list of additional_instructions describe how to interpret contract language, how to resolve ambiguity, and how to choose values. Always treat the all of the rules mentioned in the additional_instructions list as authoritative rules.
    - If the list of allowed_values are explicitly specified in the <INPUT_SCHEMA> schema, you must match the extracted value to the allowed_values.
    - If the allowed_values are specified as "ANY", you can extract any value that is relevant to the field.
    - You must strictly follow the output_format defined in the <INPUT_SCHEMA> schema and the below 9 critical rules.
        1. If output_format = "string" → return a JSON string value (not a list).
        2. If output_format = "list" → always return a JSON array.
        3. If multiple values appear for a list field, return all values as a list.
        4. If one value appears for a list field, return a single-item list.
        5. If no values appear for a list field, return an empty list ([]).
        6. Never return a string where a list is required.
        7. Never return a list where a string is required.
        8. Do not add fields that are not in the blueprint.
        9. Do not change field names. 
    - Only extract information that is explicitly visible on this page. 
    - Do NOT infer or guess values from other pages.
    - All bounding boxes MUST correspond exactly to the visible source text.
    - Bounding boxes must be in image pixel coordinates: [x1, y1, x2, y2].
    - Return structured JSON only. No explanations or commentary.
    - All fields must be present, even if there are no values found on this page, with a confidence score of 0.0.
"""

USER_PROMPT = f"""
    Analyze the following image, which represents ONE page of a media contract.

    <INPUT_SCHEMA>
    {BASE_INFO_SCHEMA}
    </INPUT_SCHEMA>

    Return all detected metadata fields as per the above <INPUT_SCHEMA> found on this page, using the below output format for each field:
    {{
        "page_has_contract_content": <boolean>,
        "extractions": {{
            "<field_name>": {{
                "value": "<exact extracted text>",
                "bbox": [x1, y1, x2, y2],
                "confidence": <float between 0.0 and 1.0>,
                "page_number": <integer>,
                "original_text": "<source text chunk that was used to extract the value>"
            }},
            ...
        }}
    }}
    
    CRITICAL RULES:
     - If no metadata fields are found on this page, return an empty dictionary for the "extractions", and set "page_has_contract_content" to False.
     - If metadata fields are found on this page, set "page_has_contract_content" to True.
     - The "original_text" key should be the exact text chunk that was used to extract the value.
     - The "bbox" key should be the bounding box co-ordinates of the original_text that was used to extract the value.
     - The "confidence" key should be a float between 0.0 and 1.0 that represents the confidence in the extracted value.
     - The "page_number" key should be the page number of the page that the text chunk was found on. Please use the page_number in the 'metadata' to get the page number.
     - The "value" key should be the exact extracted text.
     - The "field_name" key should be the name of the field that was extracted.
     - The "extractions" key should be a dictionary of the extracted values.
"""

# ============================================================================
# S3 Image Fetching
# ============================================================================

# Boto3 config for S3 operations
_s3_boto_config = BotoConfig(
    connect_timeout=5,
    read_timeout=30,
    max_pool_connections=10,
    retries={"max_attempts": 3, "mode": "standard"},
)


@lru_cache(maxsize=1)
def get_s3_client():
    """Get cached S3 client for fetching images."""
    upload_config = get_upload_config()
    region = upload_config.get("region", "us-east-1")
    return boto3.client("s3", region_name=region, config=_s3_boto_config)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI into bucket and key.

    Args:
        s3_uri: S3 URI in format s3://bucket/key

    Returns:
        Tuple of (bucket, key)
    """
    # Remove s3:// prefix
    path = s3_uri.replace("s3://", "")
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def fetch_image_from_s3(s3_uri: str) -> tuple[bytes, str]:
    """
    Fetch an image from S3 and return bytes with media type.

    Args:
        s3_uri: S3 URI of the image

    Returns:
        Tuple of (image_bytes, media_type)
    """
    s3_client = get_s3_client()
    bucket, key = parse_s3_uri(s3_uri)

    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()

    # Determine media type from key extension
    if key.lower().endswith(".jpeg") or key.lower().endswith(".jpg"):
        media_type = "image/jpeg"
    elif key.lower().endswith(".png"):
        media_type = "image/png"
    else:
        # Default to jpeg
        media_type = "image/jpeg"

    return image_bytes, media_type


# ============================================================================
# Result Aggregation
# ============================================================================


def aggregate_page_extractions(
    page_results: list[dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate page-level extractions into document-level metadata.

    For each field in the schema, pick the highest-confidence candidate
    from across all pages, preserving bbox and confidence.

    Args:
        page_results: List of extraction results from each page
        schema: The base info schema defining expected fields

    Returns:
        Aggregated metadata dictionary with full extraction info
    """
    aggregated = {}
    schema_fields = schema.get("schema", {})

    for field_name, field_config in schema_fields.items():
        output_format = field_config.get("output_format", "string")
        candidates = []

        # Collect all extractions for this field from all pages
        for result in page_results:
            if not result.get("success") or not result.get("extracted_data"):
                continue

            extractions = result["extracted_data"].get("extractions", {})
            if field_name not in extractions:
                continue

            field_data = extractions[field_name]
            # Normalize to list for uniform processing
            items = field_data if isinstance(field_data, list) else [field_data]
            candidates.extend(items)

        # Select best candidate based on output format
        if not candidates:
            aggregated[field_name] = [] if output_format == "list" else None
            continue

        if output_format == "list":
            # Find unique items by value, keeping highest confidence for each
            unique_items = {}
            for item in candidates:
                value_key = str(item.get("value"))

                # Keep item with highest confidence for each unique value
                if value_key not in unique_items or item.get(
                    "confidence", 0
                ) > unique_items[value_key].get("confidence", 0):
                    unique_items[value_key] = item

            aggregated[field_name] = list(unique_items.values())
        else:
            # For string fields, pick the dict with highest confidence
            best = max(candidates, key=lambda x: x.get("confidence", 0))
            aggregated[field_name] = best

    return aggregated


def parse_llm_response(response_text: str) -> dict[str, Any]:
    """
    Parse the LLM response text into a structured dictionary.

    Expects the LLM to return JSON format.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Parsed dictionary or empty dict if parsing fails
    """
    try:
        # Try to find JSON in the response
        # First, try direct parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    try:
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end > start:
                return json.loads(response_text[start:end].strip())
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end > start:
                return json.loads(response_text[start:end].strip())
    except json.JSONDecodeError:
        pass

    logger.warning(f"Failed to parse LLM response as JSON: {response_text[:200]}...")
    return {}


# ============================================================================
# Main Agent Node Class
# ============================================================================


class BaseInfoExtractorNode(BaseWorkflowNode):
    """
    Extract basic contract information from document images using LLM.

    This agent:
    1. Fetches images from S3 using page_images from state
    2. Uses the BASE_INFO_SCHEMA for extraction
    3. Makes parallel LLM calls (one per page image)
    4. Aggregates results and picks highest-confidence values
    """

    def __init__(self):
        super().__init__("BaseInfoExtractor")

    async def execute(self, state: WorkflowState) -> dict[str, Any]:
        """
        Extract basic contract information from document images.

        Args:
            state: Current workflow state containing page_images

        Returns:
            Updated state fields with extracted_base_info
        """
        group_id, identifier_name, document_group = self._get_context(state)
        page_images = state.get("page_images")

        self.logger.info(f"[{group_id}] Starting base info extraction")

        # Update progress
        self._update_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="base_info_starting",
            step_name="Base Info Extraction",
            status=WorkflowStatus.IN_PROGRESS,
            message="Starting contract information extraction...",
        )

        # Check if we have images to process
        if not page_images:
            error_msg = "No page images available for extraction"
            self.logger.error(f"[{group_id}] {error_msg}")

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="base_info_failed",
                step_name="Base Info Extraction",
                status=WorkflowStatus.FAILED,
                message=error_msg,
            )

            return self._create_error_response(
                step_id="base_info_failed",
                error_message=error_msg,
                extracted_base_info=None,
            )

        try:
            # Load the schema
            schema = BASE_INFO_SCHEMA

            # Initialize Bedrock client
            bedrock_client = BedrockClient()

            # Collect all page images from all file types
            all_pages = []
            for file_type, pages in page_images.items():
                for page_num, s3_uri in pages.items():
                    all_pages.append(
                        {
                            "file_type": file_type,
                            "page_number": page_num,
                            "s3_uri": s3_uri,
                        }
                    )
                    break  # Only process one page per file type for now

            total_pages = len(all_pages)
            self.logger.info(
                f"[{group_id}] Processing {total_pages} page(s) for extraction"
            )

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="base_info_fetching",
                step_name="Base Info Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message=f"Fetching {total_pages} page image(s) from storage...",
            )

            # Prepare requests for parallel processing
            requests = []
            for page_info in all_pages:
                # Fetch image from S3
                image_bytes, media_type = fetch_image_from_s3(page_info["s3_uri"])

                requests.append(
                    {
                        "user_prompt": USER_PROMPT,
                        "images": [(image_bytes, media_type)],
                        "system_prompt": (
                            SYSTEM_PROMPT if SYSTEM_PROMPT.strip() else None
                        ),
                        "metadata": {
                            "file_type": page_info["file_type"],
                            "page_number": page_info["page_number"],
                            "s3_uri": page_info["s3_uri"],
                        },
                    }
                )

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="base_info_extracting",
                step_name="Base Info Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message=f"Extracting information from {total_pages} page(s) in parallel...",
            )

            # Make parallel LLM calls in a thread pool to avoid blocking the event loop
            # This allows FastAPI to continue serving progress poll requests during extraction
            loop = asyncio.get_event_loop()
            llm_results = await loop.run_in_executor(
                None, bedrock_client.invoke_parallel, requests
            )

            # Process results
            page_extractions = []
            successful_pages = 0

            for result in llm_results:
                if result["success"]:
                    response_text = bedrock_client.extract_text_response(
                        result["response"]
                    )
                    extracted_data = parse_llm_response(response_text)

                    page_extractions.append(
                        {
                            "success": True,
                            "metadata": result["metadata"],
                            "extracted_data": extracted_data,
                        }
                    )
                    successful_pages += 1
                else:
                    page_extractions.append(
                        {
                            "success": False,
                            "metadata": result["metadata"],
                            "extracted_data": {},
                            "error": result["error"],
                        }
                    )
                    self.logger.warning(
                        f"[{group_id}] Page extraction failed for page "
                        f"{result['metadata'].get('page_number')}: {result['error']}"
                    )

            self.logger.info(
                f"[{group_id}] Successfully extracted from {successful_pages}/{total_pages} pages"
            )

            # Aggregate results
            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="base_info_aggregating",
                step_name="Base Info Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message="Aggregating extracted information...",
            )

            aggregated_info = aggregate_page_extractions(page_extractions, schema)

            self.logger.info(
                f"[{group_id}] Base info extraction completed successfully"
            )

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="base_info_complete",
                step_name="Base Info Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message="Contract information extraction completed",
            )

            return self._create_success_response(
                step_id="base_info_complete",
                extracted_base_info=aggregated_info,
            )

        except Exception as e:
            error_msg = f"Base info extraction failed: {str(e)}"
            self.logger.error(f"[{group_id}] {error_msg}")

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="base_info_failed",
                step_name="Base Info Extraction",
                status=WorkflowStatus.FAILED,
                message=error_msg,
            )

            return self._create_error_response(
                step_id="base_info_failed",
                error_message=error_msg,
                extracted_base_info=None,
            )
