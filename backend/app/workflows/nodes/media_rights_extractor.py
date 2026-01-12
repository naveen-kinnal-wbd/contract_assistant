"""
Media rights extractor agent node for LangGraph workflow.

This node extracts media rights information from contract document images using
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

USER_PROMPT = """
    <PERSONA>
    You are a highly skilled legal data analyst specializing in entertainment and media contracts.
    You are performing a PERCEPTION-ONLY task: locating and extracting relevant text from a contract page.
    You MUST NOT normalize, interpret, infer, or map values to controlled vocabularies.
    </PERSONA>

    <TASK>
    Your task is to extract RAW CONTRACT LANGUAGE for the fields defined in <INPUT_SCHEMA> schema from a SINGLE PAGE of a media contract.
    The page is provided as an image. Read the page carefully and extract the relevant text for each field defined in the <INPUT_SCHEMA> schema.
    This step is LIMITED TO TEXT EXTRACTION ONLY. All normalization, interpretation, and allowed_values matching will happen in later steps.
    Use the below <SYSTEM INSTRUCTIONS> as the authoritative set of instructions for extracting the metadata.
    </TASK>

    <INPUT_SCHEMA>
    {INPUT_SCHEMA}
    </INPUT_SCHEMA>

    <OUTPUT FORMAT>
    Return structured JSON ONLY, using the following format:

    {
        "<field_name>": {
            "context_clause": "<full contextual clause>",
            "raw_value_snippet": "<minimal value text>",
            "confidence": <float between 0.0 and 1.0>,
            "reference": { 
                "page_number": <integer>,
                "location_in_page": "<location in page>",
                "bbox": [x1, y1, x2, y2],
            }
        }
        ...
    }
    </OUTPUT FORMAT>

    <SYSTEM INSTRUCTIONS>
    - Use the description and extraction_constraints in the <INPUT_SCHEMA> schema to understand the fields and their extraction constraints. Follow the description and extraction constraints strictly.
    - Use the 'extraction_constraints' in the <INPUT_SCHEMA> schema for each field to extract the value, if they are specified. The 'extraction_constraints' are a list of instructions that describe how to find the explicit text that corresponds to the field.
    - DO NOT infer, interpret, normalize, or map values. DO NOT guess missing information.
    - Return structured JSON only. No explanations or commentary.
    - The "field_name" key in the <OUTPUT FORMAT> should be the name of the field that was extracted from the <INPUT_SCHEMA> schema.
    - The "context_clause" key in the <OUTPUT FORMAT> should be the full sentence or paragraph that contains the raw_value_snippet. It will be used for auditability and downstream reasoning.
    - The "raw_value_snippet" key in the <OUTPUT FORMAT> should be the shortest possible text span that directly expresses the value. It must be value-dense. Typically 3–20 words.
    - The "confidence" key in the <OUTPUT FORMAT> should be a float between 0.0 and 1.0 that represents the confidence in the extracted value.
    - The "bbox" key in the <OUTPUT FORMAT> should be the bounding box co-ordinates of the context_clause that was extracted. Bounding boxes must be in image pixel coordinates: [x1, y1, x2, y2].
    - The "page_number" key in the <OUTPUT FORMAT> should be the page number of the page that the text chunk was found on.
    - The "location_in_page" key in the <OUTPUT FORMAT> should be the location in the page that the text chunk was found on. It should be a string that describes the location of the text chunk in the page.
    </SYSTEM INSTRUCTIONS>
"""


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

            # Collect all page images from all file types
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

            request_body = bedrock_client.prepare_request(
                prompt=USER_PROMPT,
                images=images_list,
                schema=MEDIA_RIGHTS_SCHEMA,
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, bedrock_client.invoke, request_body
            )

            extracted_data = bedrock_client.extract_json_response(response)

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
