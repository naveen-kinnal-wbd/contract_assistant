"""
Contract processing service with async workflow management
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

from ..models.schemas import (
    DocumentGroup,
    WorkflowStatus,
    WorkflowStep,
    WorkflowProgress,
    ProcessingResponse,
)

logger = logging.getLogger(__name__)


class ContractService:
    """Service for processing contract documents asynchronously"""

    # In-memory storage for workflow progress (in production, use Redis or database)
    _workflow_store: dict[str, WorkflowProgress] = {}
    _callbacks: dict[str, list[Callable]] = {}

    @classmethod
    def get_workflow_progress(cls, group_id: str) -> Optional[WorkflowProgress]:
        """Retrieve current workflow progress for a document group"""
        return cls._workflow_store.get(group_id)

    @classmethod
    def register_callback(cls, group_id: str, callback: Callable):
        """Register a callback for workflow updates"""
        if group_id not in cls._callbacks:
            cls._callbacks[group_id] = []
        cls._callbacks[group_id].append(callback)

    @classmethod
    def _notify_callbacks(cls, group_id: str, progress: WorkflowProgress):
        """Notify all registered callbacks of progress update"""
        for callback in cls._callbacks.get(group_id, []):
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Callback error for group {group_id}: {e}")

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
    ) -> WorkflowProgress:
        """Update workflow progress with a new step"""
        step = WorkflowStep(
            step_id=step_id,
            step_name=step_name,
            status=status,
            message=message,
            timestamp=datetime.utcnow(),
            requires_feedback=requires_feedback,
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

        cls._notify_callbacks(group_id, cls._workflow_store[group_id])
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

            # Step 4: Complete
            logger.info(f"[{group_id}] Processing with blueprint refinement completed")
            final_progress = cls._update_progress(
                group_id=group_id,
                identifier_name=identifier_name,
                step_id="complete",
                step_name="Processing Complete",
                status=WorkflowStatus.COMPLETED,
                message="Blueprint refinement and extraction completed successfully.",
            )

            return ProcessingResponse(
                group_id=group_id,
                status=WorkflowStatus.COMPLETED,
                message="Blueprint refinement completed successfully.",
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
    def clear_workflow(cls, group_id: str):
        """Clear workflow data for a document group"""
        if group_id in cls._workflow_store:
            del cls._workflow_store[group_id]
        if group_id in cls._callbacks:
            del cls._callbacks[group_id]
