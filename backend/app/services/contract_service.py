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
    ProgramSelectionRequest,
    WorkflowStatus,
    WorkflowStep,
    WorkflowProgress,
    ProcessingResponse,
)
from ..workflows.graphs.blueprint_refinement import (
    run_blueprint_refinement_workflow,
    resume_workflow_with_selection,
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

    # Store for pending workflows awaiting asset selection
    _pending_selection: dict[str, DocumentGroup] = {}

    # Note: Program selection state is now managed by LangGraph's checkpointer
    # instead of manual _pending_program_selection dict

    @classmethod
    def initialize_workflow_progress(
        cls,
        group_id: str,
        identifier_name: str,
    ) -> WorkflowProgress:
        """
        Initialize workflow progress before starting background task.
        This ensures the frontend can poll for progress immediately.
        """
        step = WorkflowStep(
            step_id="initializing",
            step_name="Initializing",
            status=WorkflowStatus.IN_PROGRESS,
            message="Initializing workflow...",
            timestamp=datetime.utcnow(),
            requires_feedback=False,
            selection_data=None,
        )

        cls._workflow_store[group_id] = WorkflowProgress(
            group_id=group_id,
            identifier_name=identifier_name,
            current_status=WorkflowStatus.IN_PROGRESS,
            steps=[step],
            current_step_index=0,
        )

        return cls._workflow_store[group_id]

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
        Process contract documents with blueprint refinement using LangGraph workflow.
        This is an async method that can be run in parallel for different document groups.

        The workflow may end in AWAITING_FEEDBACK status if program selection is needed.
        In that case, the workflow state is stored for later continuation.
        """
        group_id = document_group.group_id
        identifier_name = document_group.identifier_name

        try:
            # Step 1: Upload documents
            logger.info(f"[{group_id}] Starting blueprint refinement workflow")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="upload",
                step_name="Document Upload",
                status=WorkflowStatus.IN_PROGRESS,
                message="Uploading documents to S3...",
            )

            # Run the LangGraph workflow
            final_state = await run_blueprint_refinement_workflow(document_group)

            # Check if workflow failed
            if final_state["status"] == WorkflowStatus.FAILED:
                error_msg = final_state.get("error_message", "Workflow failed")
                logger.error(f"[{group_id}] Workflow failed: {error_msg}")
                return ProcessingResponse(
                    group_id=group_id,
                    status=WorkflowStatus.FAILED,
                    message=error_msg,
                    workflow_progress=cls.get_workflow_progress(group_id),
                )

            # Check if workflow is awaiting program selection (interrupted)
            if final_state["status"] == WorkflowStatus.AWAITING_FEEDBACK:
                logger.info(
                    f"[{group_id}] Workflow interrupted - awaiting program selection"
                )
                # State is automatically persisted by LangGraph's checkpointer
                # Use resume_workflow_with_selection() to continue after user selects
                return ProcessingResponse(
                    group_id=group_id,
                    status=WorkflowStatus.AWAITING_FEEDBACK,
                    message="Program selection required. Please select a program to continue.",
                    workflow_progress=cls.get_workflow_progress(group_id),
                )

            # Workflow completed successfully without needing selection
            logger.info(f"[{group_id}] Blueprint refinement workflow completed")
            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="workflow_complete",
                step_name="Workflow Complete",
                status=WorkflowStatus.COMPLETED,
                message="Blueprint refinement completed successfully.",
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.COMPLETED,
                message="Blueprint refinement completed successfully.",
                workflow_progress=final_progress,
            )

        except Exception as e:
            logger.error(f"[{group_id}] Blueprint refinement failed: {e}")
            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="error",
                step_name="Error",
                status=WorkflowStatus.FAILED,
                message=f"Blueprint refinement failed: {str(e)}",
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.FAILED,
                message=f"Blueprint refinement failed: {str(e)}",
                workflow_progress=final_progress,
            )

    @classmethod
    async def continue_after_asset_selection(
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
    async def continue_after_program_selection(
        cls, request: ProgramSelectionRequest
    ) -> ProcessingResponse:
        """
        Continue the workflow after user selects a program.

        Uses LangGraph's interrupt resume mechanism to continue
        the workflow from where it paused at asset_selector.

        This method:
        1. Builds the selected program dict from the request
        2. Resumes the workflow using Command(resume=selected_program)
        3. The workflow continues from the interrupt point to completion
        """
        group_id = request.group_id

        # Get identifier_name from current progress if available
        identifier_name = ""
        progress = cls.get_workflow_progress(group_id)
        if progress:
            identifier_name = progress.identifier_name

        try:
            # Build the selected program dict from request
            selected_program = {
                "program_name": request.program_name,
                "contract_type": request.contract_type,
                "contract_name": request.contract_name,
                "parties": request.parties,
                "date_effective": request.date_effective,
                "date_executed": request.date_executed,
            }

            logger.info(
                f"[{group_id}] Resuming workflow with program: {request.program_name}"
            )

            # Resume the workflow with the selection using LangGraph interrupt resume
            final_state = await resume_workflow_with_selection(
                group_id=group_id,
                selected_program=selected_program,
            )

            # Return response based on final state
            if final_state.get("status") == WorkflowStatus.FAILED:
                error_msg = final_state.get("error_message", "Workflow failed")
                return ProcessingResponse(
                    group_id=group_id,
                    status=WorkflowStatus.FAILED,
                    message=error_msg,
                    workflow_progress=cls.get_workflow_progress(group_id),
                )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.COMPLETED,
                message=f"Processing completed for program: {request.program_name}",
                workflow_progress=cls.get_workflow_progress(group_id),
            )

        except Exception as e:
            logger.error(f"[{group_id}] Failed to resume workflow: {e}")
            cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="error",
                step_name="Error",
                status=WorkflowStatus.FAILED,
                message=f"Failed to resume workflow: {str(e)}",
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.FAILED,
                message=f"Failed to resume workflow: {str(e)}",
                workflow_progress=cls.get_workflow_progress(group_id),
            )

    @classmethod
    def clear_workflow(cls, group_id: str):
        """Clear workflow data for a document group"""
        if group_id in cls._workflow_store:
            del cls._workflow_store[group_id]
        if group_id in cls._pending_selection:
            del cls._pending_selection[group_id]
        # Note: LangGraph checkpointer state is not cleared here
        # In production with persistent checkpointer, you may want to clear that too
