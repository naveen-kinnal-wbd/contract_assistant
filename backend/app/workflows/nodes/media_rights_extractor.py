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
    "Media": {
        "description": "Ways the program can be exhibited, as explicitly stated in the contract.",
        "extraction_constraints": [
            "Extract only clauses that grant or limit exhibition or distribution rights.",
            "The clause must describe how the content may be shown, transmitted, streamed, or made available to audiences.",
            "Look for references to delivery modes such as television, streaming services, video-on-demand, physical media, or similar mechanisms.",
            "Do NOT extract clauses that only describe marketing, promotion, technical delivery, or definitions without granting rights.",
        ],
    },
    "Outlet": {
        "description": "Specific networks, brands, platforms, services, or affiliated outlets that are granted exhibition or distribution rights under the contract.",
        "extraction_constraints": [
            "Extract only clauses that explicitly grant, restrict, or define which networks, channels, platforms, brands, or affiliated outlets may exhibit the program.",
            "The clause must identify one or more outlets by name, brand, service, platform, or organizational grouping (e.g., specific channels, streaming services, digital platforms, or affiliated networks).",
            "Look for named entities that correspond to television networks, streaming platforms, digital services, branded channels, or corporate outlet groups, even if phrased broadly.",
            "Include clauses that grant rights to all outlets owned, operated, or controlled by a company, or to all current and future affiliated outlets.",
            "Capture both explicitly named outlets and umbrella or catch-all language (e.g., 'all affiliated networks', 'any platform owned by the Company', 'all branded services worldwide').",
            "Do NOT extract clauses that only describe geographic territory, media type, marketing obligations, technical delivery methods, or internal business units unless they explicitly grant exhibition rights to identifiable outlets.",
        ],
    },
    "Term Start": {
        "description": "Date or legally operative phrase indicating when the granted rights for the applicable rights package first become effective.",
        "extraction_constraints": [
            "Extract only the clause or phrase that explicitly defines when the granted rights begin to apply for the specific rights package described.",
            "Look for legal trigger language such as 'as of', 'commencement date', 'effective date', 'beginning on', 'from and after', or equivalent formulations.",
            "The extracted text must describe the start of rights applicability, not merely the execution, signing, or delivery date unless the clause explicitly ties those events to rights commencement.",
            "If the start of rights is conditional (e.g., 'upon delivery', 'upon first exhibition', 'upon acceptance'), extract the full conditional phrase rather than inferring or normalizing a calendar date.",
            "If a specific calendar date is stated, preserve it exactly as written in the source text.",
            "Do NOT extract dates that relate solely to payment schedules, production milestones, audit periods, or unrelated contractual obligations.",
            "Do NOT infer a term start date if none is explicitly stated in relation to the rights grant.",
        ],
    },
    "Term End": {
        "description": "Date, duration, or legally operative phrase indicating when the granted rights for the applicable rights package expire or terminate.",
        "extraction_constraints": [
            "Extract only the clause or phrase that explicitly defines when the granted rights end, expire, or terminate for the specific rights package described.",
            "Look for legal language such as 'until', 'through', 'expiration date', 'termination date', 'for a period of', 'for X years', or equivalent formulations.",
            "If the end of rights is expressed as a duration (e.g., 'for five (5) years from commencement'), extract the full duration phrase exactly as written.",
            "If rights are granted without a defined end date using language such as 'in perpetuity', 'in perpetuum', or 'forever', extract the exact phrase used in the contract.",
            "If the end of rights is conditional (e.g., 'until termination of the agreement', 'until licensee ceases operations'), extract the full conditional phrase.",
            "Do NOT normalize durations into calculated dates at this stage.",
            "Do NOT extract termination provisions, breach clauses, or general contract expiration language unless they explicitly govern the expiration of the granted rights.",
        ],
    },
    "Territories": {
        "description": "Geographical locations or territorial scopes where the granted rights may be exercised, as explicitly stated in the contract.",
        "extraction_constraints": [
            "Extract only clauses that explicitly define, limit, or describe the geographic scope of the granted rights.",
            "The clause must specify where the program may or may not be exhibited, distributed, or exploited geographically.",
            "Look for explicit mentions of countries, regions, groupings of countries, or global scope terms (e.g., 'worldwide', 'international', 'domestic', 'outside the Territory').",
            "Preserve the territorial language exactly as written, including lists of countries, regional groupings, exclusions, inclusions, or carve-outs.",
            "Include clauses that define territories by reference to political, economic, or contractual groupings (e.g., 'European Union', 'Latin America', 'Middle East').",
            "Include both inclusive and exclusive territorial statements (e.g., 'worldwide excluding...', 'except for...', 'limited to...').",
            "Do NOT infer, expand, or collapse territories beyond what is explicitly stated in the clause.",
            "Do NOT extract clauses that mention geographic locations solely for purposes of production, delivery logistics, governing law, marketing activities, or venue unless they explicitly define rights territory.",
        ],
    },
    "Venues": {
        "description": "Physical locations, access environments, or audience-facing contexts in which end users are permitted to access or experience the content, as explicitly stated in the contract.",
        "extraction_constraints": [
            "Extract only clauses that explicitly define, limit, or describe the permitted venues, access environments, or audience contexts for exercising the granted rights.",
            "The clause must specify where, how, or in what type of environment end users may view or access the content (e.g., institutional settings, transportation, retail locations, subscriber-based access).",
            "Look for explicit references to physical venues (e.g., schools, hospitals, museums, airlines), institutional environments (e.g., educational, governmental, military), or access-based contexts (e.g., affiliate subscribers, direct-to-consumer).",
            "Include clauses that group venues into broader categories (e.g., 'institutional use', 'transportation venues', 'retail outlets') if explicitly stated.",
            "Include both inclusive and exclusive venue language (e.g., 'non-theatrical only', 'excluding retail', 'limited to educational institutions').",
            "Preserve the venue language exactly as written, including lists, groupings, exclusions, and carve-outs.",
            "Do NOT infer, expand, or collapse venues beyond what is explicitly stated in the clause.",
            "Do NOT extract clauses that mention venues solely for marketing, promotion, internal business operations, technical delivery, or definitions unless they explicitly define rights usage.",
        ],
    },
    "Languages": {
        "description": "Languages or linguistic forms in which the content may be exhibited to end users, as explicitly permitted or restricted by the contract.",
        "extraction_constraints": [
            "Extract only clauses that explicitly grant, restrict, or describe the permitted languages or linguistic versions in which the content may be exhibited as part of the granted rights.",
            "The clause must define allowable exhibition languages, including original language, dubbed versions, subtitled versions, voice-over, or combinations thereof.",
            "Look for explicit references to specific languages (e.g., English, Spanish), groups of languages (e.g., local languages, all languages), or conditional language permissions (e.g., 'if available', 'subject to availability').",
            "Preserve any qualifiers related to language format, such as 'dubbed', 'subtitled', 'voice-over', 'original language', or 'no dialogue'.",
            "Include clauses that limit languages by territory or outlet if explicitly stated.",
            "Preserve exclusionary language (e.g., 'excluding English', 'non-English languages only').",
            "Do NOT infer 'All Languages' unless the clause explicitly states 'all languages', 'any language', or equivalent language granting.",
            "Do NOT extract clauses that reference language solely for technical delivery, production requirements, QC specifications, or definitions unless they explicitly govern exhibition rights.",
        ],
    },
}

USER_PROMPT = """
    <PERSONA>
    You are a highly skilled legal data analyst specializing in entertainment and media contracts.
    You are performing a PERCEPTION-ONLY task: locating and extracting relevant text from a contract page.
    You MUST NOT normalize, interpret, infer, or map values to controlled vocabularies.
    </PERSONA>

    <TASK>
    Your task is to extract RAW CONTRACT LANGUAGE for the fields defined in <INPUT_SCHEMA> from a SINGLE PAGE of a media contract.
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
    - The "field_name" key in the <OUTPUT FORMAT> should be the name of the field that was extracted from the <INPUT_SCHEMA> schema.
    - The "context_clause" key in the <OUTPUT FORMAT> should be the full sentence or paragraph that contains the raw_value_snippet. It will be used for auditability and downstream reasoning.
    - The "raw_value_snippet" key in the <OUTPUT FORMAT> should be the shortest possible text span that directly expresses the value. It must be value-dense. Typically 3–20 words.
    - The "confidence" key in the <OUTPUT FORMAT> should be a float between 0.0 and 1.0 that represents the confidence in the extracted value.
    - The "bbox" key in the <OUTPUT FORMAT> should be the bounding box co-ordinates of the context_clause that was extracted. Bounding boxes must be in image pixel coordinates: [x1, y1, x2, y2].
    - The "page_number" key in the <OUTPUT FORMAT> should be the page number of the page that the text chunk was found on.
    - The "location_in_page" key in the <OUTPUT FORMAT> should be the location in the page that the text chunk was found.
    - All string values MUST escape special characters, including Double quotes, Newlines, and Tabs.
    - All string values MUST be escaped for output JSON.
    - Return structured JSON only. No explanations or commentary.
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
