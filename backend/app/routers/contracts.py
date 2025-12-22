"""
Contract processing API endpoints
"""

import asyncio
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import json

from ..models.schemas import (
    AssetSelectionRequest,
    DocumentGroup,
    WorkflowProgress,
    ProcessingResponse,
    WorkflowStatus,
)
from ..services.contract_service import ContractService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/contracts", tags=["contracts"])


@router.post("/processContractInference", response_model=ProcessingResponse)
async def process_contract_inference(
    document_group: DocumentGroup,
    background_tasks: BackgroundTasks,
):
    """
    Process contract documents without blueprint refinement.

    This endpoint initiates async processing and returns immediately with a tracking ID.
    Use the /progress/{group_id} endpoint to track processing status.
    """
    logger.info(
        f"Received contract inference request for group: {document_group.group_id}"
    )

    # Initialize progress immediately so frontend can poll it right away
    initial_progress = ContractService.initialize_workflow_progress(
        group_id=document_group.group_id,
        identifier_name=document_group.identifier_name,
    )

    # Start processing in background
    background_tasks.add_task(
        ContractService.process_contract_inference,
        document_group,
    )

    return ProcessingResponse(
        group_id=document_group.group_id,
        status=WorkflowStatus.IN_PROGRESS,
        message="Contract inference processing started.",
        workflow_progress=initial_progress,
    )


@router.post("/processBlueprintsRefinement", response_model=ProcessingResponse)
async def process_blueprints_refinement(
    document_group: DocumentGroup,
    background_tasks: BackgroundTasks,
):
    """
    Process contract documents with blueprint refinement.

    This endpoint initiates async processing and returns immediately with a tracking ID.
    Use the /progress/{group_id} endpoint to track processing status.
    """
    logger.info(
        f"Received blueprint refinement request for group: {document_group.group_id}"
    )

    # Initialize progress immediately so frontend can poll it right away
    initial_progress = ContractService.initialize_workflow_progress(
        group_id=document_group.group_id,
        identifier_name=document_group.identifier_name,
    )

    # Start processing in background
    background_tasks.add_task(
        ContractService.process_blueprints_refinement,
        document_group,
    )

    return ProcessingResponse(
        group_id=document_group.group_id,
        status=WorkflowStatus.IN_PROGRESS,
        message="Blueprint refinement processing started.",
        workflow_progress=initial_progress,
    )


@router.get("/progress/{group_id}", response_model=Optional[WorkflowProgress])
async def get_workflow_progress(group_id: str):
    """
    Get the current workflow progress for a document group.
    """
    progress = ContractService.get_workflow_progress(group_id)
    if progress is None:
        raise HTTPException(
            status_code=404, detail=f"No workflow found for group ID: {group_id}"
        )
    return progress


@router.post("/select-asset", response_model=ProcessingResponse)
async def select_asset(request: AssetSelectionRequest):
    """
    Submit asset selection to continue the workflow.
    """
    logger.info(f"Received asset selection for group: {request.group_id}")

    response = await ContractService.continue_after_asset_selection(request)

    return response


@router.get("/progress/{group_id}/stream")
async def stream_workflow_progress(group_id: str):
    """
    Stream workflow progress updates using Server-Sent Events (SSE).
    """

    async def event_generator():
        last_step_count = 0
        max_wait_iterations = 60  # Max 60 seconds of waiting
        iterations = 0

        while iterations < max_wait_iterations:
            progress = ContractService.get_workflow_progress(group_id)

            if progress:
                current_step_count = len(progress.steps)

                # Send update if there are new steps
                if current_step_count > last_step_count:
                    last_step_count = current_step_count
                    yield f"data: {progress.model_dump_json()}\n\n"

                # Check if workflow is complete or failed
                if progress.current_status in [
                    WorkflowStatus.COMPLETED,
                    WorkflowStatus.FAILED,
                ]:
                    yield f"data: {progress.model_dump_json()}\n\n"
                    break

            await asyncio.sleep(0.5)
            iterations += 1

        yield 'data: {"event": "close"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.delete("/progress/{group_id}")
async def clear_workflow_progress(group_id: str):
    """
    Clear workflow data for a document group.
    """
    ContractService.clear_workflow(group_id)
    return {"message": f"Workflow data cleared for group: {group_id}"}


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "contract-assistance-api"}
