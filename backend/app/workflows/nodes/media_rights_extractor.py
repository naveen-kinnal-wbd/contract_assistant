"""
Media rights extractor agent node for LangGraph workflow.

This node extracts media rights information from contract document images using
AWS Bedrock LLM.
"""

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from config import get_upload_config
from models.schemas import WorkflowStatus
from services.bedrock_client import BedrockClient
from workflows.state import WorkflowState
from workflows.nodes.base import BaseWorkflowNode

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPTS - Define system and user prompts for media rights extraction
# ============================================================================

MEDIA_RIGHTS_SCHEMA = {
    "schema": {
        "Media": {
            "description": "Ways the program can be exhibited, as explicitly stated in the contract.",
            "extraction_constraints": [],
        },
        "Outlet": {
            "description": "Specific networks, brands, platforms, or services granted exhibition rights.",
            "extraction_constraints": [
                "Look for named entities like HBO Max, Turner Networks, or broader categories like 'all WarnerMedia platforms.' If no specific outlets are limited and broad rights are granted to the Company, indicate 'All Company Outlets' or list specifically mentioned outlets."
            ],
        },
        "Term Start": {
            "description": "Date or date phrase indicating when rights begin.",
            "extraction_constraints": [
                "Extract the first day when the granted rights become applicable for THIS SPECIFIC rights package. Look for phrases like 'as of', 'commencement date', 'effective date', or 'beginning on'. Format should be MM/DD/YYYY if specific dates are provided. For production agreements, this may correspond to the date of the agreement unless otherwise specified."
            ],
        },
        "Term End": {
            "description": "Date, duration, or phrase indicating when rights end.",
            "extraction_constraints": [
                "Extract the last day when the granted rights remain applicable for THIS SPECIFIC rights package. Look for phrases like 'until', 'expiration date', 'termination date', or language that specifies the duration of rights. For rights granted 'in perpetuity', indicate as 'Perpetuity' to reflect no end date. Format should be MM/DD/YYYY if specific dates are provided."
            ],
        },
        "Territories": {
            "description": "Geographical regions where rights apply, as explicitly stated in the contract.",
            "extraction_constraints": [
                "Extract only the specific geographical locations explicitly mentioned in the text. Do not infer or add broader regions (e.g., do NOT output continents or regions such as 'North America', 'Europe', 'Asia') unless the locations are not explicitly mentioned. Look for specific countries, regions, or terms like 'worldwide', 'global', 'domestic' or 'international'. If rights are granted globally without territorial restrictions, indicate 'Worldwide'."
            ],
        },
        "Venues": {
            "description": "Distribution channels and places where end-users may access the content.",
            "extraction_constraints": [
                "Extract information about where and how end users can access the content. Look for terms like 'affiliate subscribers', 'direct-to-consumer', 'theatrical', 'non-theatrical', 'commercial', 'residential', 'institutional' or similar distribution channels. If comprehensive rights are granted without venue restrictions, indicate 'All Venues'."
            ],
        },
        "Languages": {
            "description": "Languages permitted for exhibition.",
            "extraction_constraints": [
                "Extract the text snippet that describes the specific languages in which the content may be exhibited, including dubbed or subtitled versions. Look for language restrictions or specifications like 'English-language', 'local language versions', or 'all languages'. If no language restrictions are mentioned with broad rights granted, indicate 'All Languages'."
            ],
        },
    }
}

USER_PROMPT = f"""
    <PERSONA>
    You are a highly skilled legal data analyst specializing in entertainment and media contracts.
    You are performing a PERCEPTION-ONLY task: locating and extracting relevant text from a contract page.
    You MUST NOT normalize, interpret, infer, or map values to controlled vocabularies.
    </PERSONA>

    <TASK>
    Your task is to extract RAW CONTRACT LANGUAGE for the fields defined in <INPUT_SCHEMA> schema from a SINGLE PAGE of a media contract.
    The page is provided as an image. Read the page carefully and extract the relevant text for each field defined in the <INPUT_SCHEMA> schema.

    This step is LIMITED TO TEXT EXTRACTION ONLY.
    All normalization, interpretation, and allowed_values matching will happen in later steps.
    </TASK>

    <INPUT_SCHEMA>
    {json.dumps(MEDIA_RIGHTS_SCHEMA, indent=4)}
    </INPUT_SCHEMA>

    <CRITICAL EXTRACTION PRINCIPLES>
    1. Use the description and extraction_constraints in the <INPUT_SCHEMA> schema to understand the fields and their extraction constraints. Follow the description and extraction constraints strictly.
    2. Use the 'extraction_constraints' in the <INPUT_SCHEMA> schema for each field to extract the value, if they are specified. The 'extraction_constraints' are a list of instructions that describe how to find the explicit text that corresponds to the field.
    3. DO NOT infer, interpret, normalize, or map values.
    4. DO NOT guess missing information.
    5. DO NOT combine information across pages.
    6. If a field is not explicitly mentioned, do not extract it.
    </CRITICAL EXTRACTION PRINCIPLES>

    <FOR EACH FIELD YOU EXTRACT>
    You MUST extract TWO distinct text spans:

    1. raw_value_snippet:
    - The SHORTEST possible text span that directly expresses the value.
    - Must be value-dense.
    - Typically 3–20 words.

    2. context_clause:
    - The full sentence or paragraph that contains the raw_value_snippet.
    - Used for auditability and downstream reasoning.

    3. bbox:
    - The bounding box co-ordinates of the context_clause that was used to extract the value.
    - Bounding boxes must be in image pixel coordinates: [x1, y1, x2, y2].

    4. confidence:
    - A float between 0.0 and 1.0 that represents the confidence in the extracted value.

    5. page_number:
    - The page number of the page that the text chunk was found on. Please use the page_number in the 'metadata' to get the page number.
    </FOR EACH FIELD YOU EXTRACT>

    <OUTPUT FORMAT>
    Return structured JSON ONLY, using the following format:

    {{
        "page_has_contract_content": <boolean>,
        "extractions": {{
            "<field_name>": {{
                "raw_value_snippet": "<minimal value text>",
                "context_clause": "<full contextual clause>",
                "bbox": [x1, y1, x2, y2],
                "confidence": <float between 0.0 and 1.0>,
                "page_number": <integer>
            }}
        }}
    }}

    <OUTPUT RULES>
        - If no relevant contract content exists on the page:
            - Set "page_has_contract_content" to false
            - Return an empty "extractions" object
        - If a field does not appear, omit it from "extractions"
        - Bounding boxes MUST exactly match visible source text provided in the context_clause key.
        - Confidence reflects certainty that the extracted text corresponds to the field (not correctness of meaning)
        - Do NOT include explanations or commentary
    </OUTPUT RULES>
"""

# ============================================================================
# S3 Image Fetching
# ============================================================================

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
    """Parse an S3 URI into bucket and key."""
    path = s3_uri.replace("s3://", "")
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def fetch_image_from_s3(s3_uri: str) -> dict[str, Any]:
    """
    Fetch an image from S3 and return image data with metadata.

    The metadata (original_filename, page_number, media_type, file_type)
    is retrieved from the S3 object's user-defined metadata.

    Args:
        s3_uri: S3 URI of the image

    Returns:
        Dict with keys: bytes, media_type, filename, page
    """
    s3_client = get_s3_client()
    bucket, key = parse_s3_uri(s3_uri)

    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()

    # Get user-defined metadata from S3 object
    s3_metadata = response.get("Metadata", {})

    # Use metadata from S3, with fallbacks
    media_type = s3_metadata.get("media_type")
    identifier_name = s3_metadata.get("identifier_name", "")
    page_number = s3_metadata.get("page_number", "")

    return {
        "bytes": image_bytes,
        "media_type": media_type,
        "identifier_name": identifier_name,
        "page_number": page_number,
    }


# ============================================================================
# Response Parsing
# ============================================================================


def parse_llm_response(response_text: str) -> dict[str, Any]:
    """Parse the LLM response text into a structured dictionary."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

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


class MediaRightsExtractorNode(BaseWorkflowNode):
    """
    Extract media rights information from contract document images using LLM.

    This agent:
    1. Fetches images from S3 using page_images from state
    2. Uses the MEDIA_RIGHTS_SCHEMA for extraction
    3. Makes a single LLM call with all page images
    4. Parses and returns the extracted data
    """

    def __init__(self):
        super().__init__("MediaRightsExtractor")

    async def execute(self, state: WorkflowState) -> dict[str, Any]:
        """
        Extract media rights information from contract document images.

        Args:
            state: Current workflow state containing page_images

        Returns:
            Updated state fields with extracted_media_rights
        """
        group_id, identifier_name, document_group = self._get_context(state)
        page_images = state.get("page_images")

        self.logger.info(f"[{group_id}] Starting media rights extraction")

        self._update_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="media_rights_starting",
            step_name="Media Rights Extraction",
            status=WorkflowStatus.IN_PROGRESS,
            message="Starting media rights extraction...",
        )

        if not page_images:
            error_msg = "No page images available for extraction"
            self.logger.error(f"[{group_id}] {error_msg}")

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="media_rights_failed",
                step_name="Media Rights Extraction",
                status=WorkflowStatus.FAILED,
                message=error_msg,
            )

            return self._create_error_response(
                step_id="media_rights_failed",
                error_message=error_msg,
                extracted_media_rights=None,
            )

        try:
            bedrock_client = BedrockClient()

            # Collect all page images from all file types (metadata from S3)
            images_list = []
            for file_type, pages in page_images.items():
                for page_num, s3_uri in pages.items():
                    image_data = fetch_image_from_s3(s3_uri)
                    images_list.append(image_data)

            total_pages = len(images_list)
            self.logger.info(
                f"[{group_id}] Processing {total_pages} page(s) for media rights extraction"
            )

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="media_rights_extracting",
                step_name="Media Rights Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message=f"Extracting media rights from {total_pages} page(s)...",
            )

            # Prepare and invoke the request
            request_body = bedrock_client.prepare_request(
                prompt=USER_PROMPT,
                images=images_list,
            )

            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, bedrock_client.invoke, request_body
            )

            # Parse response
            response_text = bedrock_client.extract_text_response(response)
            extracted_data = parse_llm_response(response_text)

            self.logger.info(
                f"[{group_id}] Media rights extraction completed successfully"
            )

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="media_rights_complete",
                step_name="Media Rights Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message="Media rights extraction completed",
            )

            return self._create_success_response(
                step_id="media_rights_complete",
                extracted_media_rights=extracted_data,
            )

        except Exception as e:
            error_msg = f"Media rights extraction failed: {str(e)}"
            self.logger.error(f"[{group_id}] {error_msg}")

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="media_rights_failed",
                step_name="Media Rights Extraction",
                status=WorkflowStatus.FAILED,
                message=error_msg,
            )

            return self._create_error_response(
                step_id="media_rights_failed",
                error_message=error_msg,
                extracted_media_rights=None,
            )
