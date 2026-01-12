"""
Base info extractor agent node for LangGraph workflow.

This node extracts basic contract metadata from document images using
AWS Bedrock LLM.
"""

import asyncio
import json
import logging
from typing import Any

from models.schemas import WorkflowStatus
from services.bedrock_client import BedrockClient
from services.s3_client import fetch_image_from_s3
from workflows.nodes.base import BaseWorkflowNode
from workflows.state import WorkflowState

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

USER_PROMPT = """
    <PERSONA>
    You are a highly skilled legal data analyst specializing in entertainment and media contracts. With your extensive background in contract law and the entertainment industry, you understand the nuances of rights agreements, licensing terms, and media distribution contracts.
    </PERSONA>

    <TASK>
    Your task is to extract metadata for the fields in the <INPUT_SCHEMA> from a SINGLE PAGE of a media contract. The page is provided as an image. 
    Take the provided JSON Schema <INPUT_SCHEMA> as the authoritative source to understand the fields, its description, its output format, its additional instructions, and its allowed values.
    Use the below <SYSTEM INSTRUCTIONS> as the authoritative set of instructions for extracting the metadata.
    </TASK>


    <INPUT_SCHEMA>
    {INPUT_SCHEMA}
    </INPUT_SCHEMA>

    Return all detected metadata fields as per the above <INPUT_SCHEMA> found on this page, using the below output format for each field:
    {
        "page_has_contract_content": <boolean>,
        "extractions": {
            "<field_name>": {
                "value": "<exact extracted text>",
                "bbox": [x1, y1, x2, y2],
                "confidence": <float between 0.0 and 1.0>,
                "page_number": <integer>,
                "original_text": "<source text chunk that was used to extract the value>"
            },
            ...
        }
    }
    
    <SYSTEM INSTRUCTIONS>
    - Extract metadata for the fields in the <INPUT_SCHEMA> schema only. Do not add or change fields that are not in the <INPUT_SCHEMA> schema.
    - When extracting metadata, you must follow the list of additional_instructions for each JSON field that is specified in the <INPUT_SCHEMA> schema, if it exists. The list of additional_instructions describe how to interpret contract language, how to resolve ambiguity, and how to choose values. Always treat the all of the rules mentioned in the additional_instructions list as authoritative rules.
    - If the list of allowed_values are explicitly specified in the <INPUT_SCHEMA> schema for a JSON field, you must match the extracted values to the allowed_values. Treat the allowed_values in the <INPUT_SCHEMA> schema as the authoritative set of values for the field.
    - Do not extract values that are not in the allowed_values for a JSON field, unless the allowed_values are specified as "ANY".
    - If the allowed_values are specified as "ANY", you can extract any value that is relevant to the field.
    - All bounding boxes MUST correspond exactly to the visible source text. Bounding boxes must be in image pixel coordinates: [x1, y1, x2, y2].
    - Return structured JSON only. No explanations or commentary.
    - If no metadata fields are found on this page, return an empty dictionary for the "extractions", and set "page_has_contract_content" to False.
    - The "original_text" key should be the paragraph of text that was used to extract the value, that provides the most context for the extracted value.
    - The "bbox" key should be the bounding box co-ordinates of the original_text that was used to extract the value.
    - The "confidence" key should be a float between 0.0 and 1.0 that represents the confidence in the extracted value.
    - The "page_number" key should be the page number of the page that the text chunk was found on. Please use the page_number in the 'metadata' to get the page number.
    - The "value" key should be the exact extracted text.
    - The "field_name" key should be the name of the field that was extracted.
    - Finally, in addition to the above critical rules, You must strictly follow the output_format defined in the <INPUT_SCHEMA> schema and the below 5 output format rules.
        1. If output_format = "string" → return a JSON string value (not a list).
        2. If output_format = "list" → always return a JSON array.
        3. If multiple values appear for a list field, return all values as a list.
        4. If one value appears for a list field, return a single-item list.
        5. If no values appear for a list field, return an empty list ([]).
    </SYSTEM INSTRUCTIONS>
"""


class BaseInfoExtractorNode(BaseWorkflowNode):
    """
    Extract basic contract information from document images using LLM.

    This agent:
    1. Fetches images from S3 using page_images from state
    2. Uses the BASE_INFO_SCHEMA for extraction
    3. Makes a single LLM call with all page images
    4. Parses and returns the extracted data
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

        self._update_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="base_info_starting",
            step_name="Base Info Extraction",
            status=WorkflowStatus.IN_PROGRESS,
            message="Starting contract information extraction...",
        )

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
            bedrock_client = BedrockClient()

            # Collect all page images from all file types
            images_list = []
            for file_type, pages in page_images.items():
                for page_num, s3_uri in pages.items():
                    image_data = fetch_image_from_s3(s3_uri)
                    images_list.append(image_data)

            total_pages = len(images_list)
            self.logger.info(
                f"[{group_id}] Processing {total_pages} page(s) for extraction"
            )

            self._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="base_info_extracting",
                step_name="Base Info Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message=f"Extracting information from {total_pages} page(s)...",
            )

            request_body = bedrock_client.prepare_request(
                prompt=USER_PROMPT,
                images=images_list,
                schema=BASE_INFO_SCHEMA,
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, bedrock_client.invoke, request_body
            )

            extracted_data = bedrock_client.extract_json_response(response)

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
                extracted_base_info=extracted_data,
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
