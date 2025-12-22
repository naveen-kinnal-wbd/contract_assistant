"""
Contract processing service with async workflow management
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models.schemas import (
    AssetSelectionRequest,
    DocumentGroup,
    WorkflowStatus,
    WorkflowStep,
    WorkflowProgress,
    ProcessingResponse,
)

logger = logging.getLogger(__name__)

# Load asset selection data from JSON file
SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples"
ASSET_SELECTION_FILE = SAMPLES_DIR / "asset_selection.json"


def load_asset_selection_data() -> list[dict]:
    """Load asset selection options from JSON file"""
    try:
        with open(ASSET_SELECTION_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load asset selection data: {e}")
        return []


class ContractService:
    """Service for processing contract documents asynchronously"""

    # In-memory storage for workflow progress (in production, use Redis or database)
    _workflow_store: dict[str, WorkflowProgress] = {}

    @classmethod
    def get_workflow_progress(cls, group_id: str) -> Optional[WorkflowProgress]:
        """Retrieve current workflow progress for a document group"""
        return cls._workflow_store.get(group_id)

    # Store for pending workflows awaiting selection
    _pending_selection: dict[str, DocumentGroup] = {}

    @classmethod
    def _update_progress(
        cls,
        group_id: str,
        identifier_name: str,
        step_id: str,
        step_name: str,
        status: WorkflowStatus,
        message: str,
        requires_feedback: bool = False,
        selection_data: Optional[list[dict]] = None,
    ) -> WorkflowProgress:
        """Update workflow progress with a new step"""
        step = WorkflowStep(
            step_id=step_id,
            step_name=step_name,
            status=status,
            message=message,
            timestamp=datetime.utcnow(),
            requires_feedback=requires_feedback,
            selection_data=selection_data,
        )

        if group_id not in cls._workflow_store:
            cls._workflow_store[group_id] = WorkflowProgress(
                group_id=group_id,
                identifier_name=identifier_name,
                current_status=status,
                steps=[step],
                current_step_index=0,
            )
        else:
            progress = cls._workflow_store[group_id]
            progress.steps.append(step)
            progress.current_status = status
            progress.current_step_index = len(progress.steps) - 1

        return cls._workflow_store[group_id]

    @classmethod
    async def process_contract_inference(
        cls, document_group: DocumentGroup
    ) -> ProcessingResponse:
        """
        Process contract documents without blueprint refinement.
        This is an async method that can be run in parallel for different document groups.
        """
        group_id = document_group.group_id
        identifier_name = document_group.identifier_name

        try:
            # Step 1: Upload documents
            logger.info(f"[{group_id}] Uploading contract documents")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="upload",
                step_name="Document Upload",
                status=WorkflowStatus.IN_PROGRESS,
                message="Uploading contract documents...",
            )

            # Simulate upload process
            await asyncio.sleep(2)

            logger.info(f"[{group_id}] Contract documents uploaded")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="upload_complete",
                step_name="Document Upload",
                status=WorkflowStatus.IN_PROGRESS,
                message="Contract documents uploaded successfully.",
            )

            # Step 2: Metadata extraction (simulated)
            logger.info(f"[{group_id}] Extracting metadata")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="extraction",
                step_name="Metadata Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message="Extracting contract metadata...",
            )

            await asyncio.sleep(2)

            # Step 3: Complete
            logger.info(f"[{group_id}] Processing completed")
            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="complete",
                step_name="Processing Complete",
                status=WorkflowStatus.COMPLETED,
                message="Contract inference completed successfully.",
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.COMPLETED,
                message="Contract inference completed successfully.",
                workflow_progress=final_progress,
            )

        except Exception as e:
            logger.error(f"[{group_id}] Processing failed: {e}")
            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="error",
                step_name="Error",
                status=WorkflowStatus.FAILED,
                message=f"Processing failed: {str(e)}",
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.FAILED,
                message=f"Processing failed: {str(e)}",
                workflow_progress=final_progress,
            )

    @classmethod
    async def process_blueprints_refinement(
        cls, document_group: DocumentGroup
    ) -> ProcessingResponse:
        """
        Process contract documents with blueprint refinement.
        This is an async method that can be run in parallel for different document groups.
        """
        group_id = document_group.group_id
        identifier_name = document_group.identifier_name

        try:
            # Step 1: Upload documents
            logger.info(f"[{group_id}] Uploading contract documents")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="upload",
                step_name="Document Upload",
                status=WorkflowStatus.IN_PROGRESS,
                message="Uploading contract documents...",
            )

            # Simulate upload process
            await asyncio.sleep(2)

            logger.info(f"[{group_id}] Contract documents uploaded")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="upload_complete",
                step_name="Document Upload",
                status=WorkflowStatus.IN_PROGRESS,
                message="Contract documents uploaded successfully.",
            )

            # Step 2: Blueprint analysis
            logger.info(f"[{group_id}] Analyzing blueprints")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="blueprint_analysis",
                step_name="Blueprint Analysis",
                status=WorkflowStatus.IN_PROGRESS,
                message="Analyzing and refining blueprints...",
            )

            await asyncio.sleep(2)

            # Step 3: Metadata extraction
            logger.info(f"[{group_id}] Extracting metadata with refined blueprints")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="extraction",
                step_name="Metadata Extraction",
                status=WorkflowStatus.IN_PROGRESS,
                message="Extracting metadata using refined blueprints...",
            )

            await asyncio.sleep(2)

            # Step 4: Asset Selection (pause for user input)
            logger.info(f"[{group_id}] Awaiting asset selection")
            asset_options = load_asset_selection_data()
            cls._pending_selection[group_id] = document_group

            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="asset_selection",
                step_name="Asset Selection",
                status=WorkflowStatus.AWAITING_FEEDBACK,
                message="Please select an asset to continue.",
                requires_feedback=True,
                selection_data=asset_options,
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.AWAITING_FEEDBACK,
                message="Awaiting asset selection.",
                workflow_progress=final_progress,
            )

        except Exception as e:
            logger.error(f"[{group_id}] Processing failed: {e}")
            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="error",
                step_name="Error",
                status=WorkflowStatus.FAILED,
                message=f"Processing failed: {str(e)}",
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.FAILED,
                message=f"Processing failed: {str(e)}",
                workflow_progress=final_progress,
            )

    @classmethod
    async def continue_after_selection(
        cls, request: AssetSelectionRequest
    ) -> ProcessingResponse:
        """
        Continue the workflow after user selects an asset.
        """
        group_id = request.group_id
        document_group = cls._pending_selection.get(group_id)
        if not document_group:
            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.FAILED,
                message="No pending selection found for this group.",
            )

        identifier_name = document_group.identifier_name

        try:
            # Mark selection received
            logger.info(
                f"[{group_id}] Asset selected: Deal {request.deal_id}, Asset {request.asset_id}"
            )
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="selection_received",
                step_name="Asset Selection",
                status=WorkflowStatus.IN_PROGRESS,
                message=f"Asset selected: Deal {request.deal_id}, Asset {request.asset_id}",
            )

            await asyncio.sleep(1)

            # Complete the workflow
            logger.info(f"[{group_id}] Processing completed with selected asset")
            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="complete",
                step_name="Processing Complete",
                status=WorkflowStatus.COMPLETED,
                message=f"Blueprint refinement and extraction completed successfully for Deal {request.deal_id} and Asset {request.asset_id}.",
            )

            # Clean up pending selection
            del cls._pending_selection[group_id]

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.COMPLETED,
                message=f"Processing completed for Deal {request.deal_id} and Asset {request.asset_id}.",
                workflow_progress=final_progress,
            )

        except Exception as e:
            logger.error(f"[{group_id}] Failed after selection: {e}")
            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="error",
                step_name="Error",
                status=WorkflowStatus.FAILED,
                message=f"Processing failed: {str(e)}",
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.FAILED,
                message=f"Processing failed: {str(e)}",
                workflow_progress=final_progress,
            )

    @classmethod
    def clear_workflow(cls, group_id: str):
        """Clear workflow data for a document group"""
        if group_id in cls._workflow_store:
            del cls._workflow_store[group_id]
        if group_id in cls._pending_selection:
            del cls._pending_selection[group_id]
