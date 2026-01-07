"""
Blueprint refinement workflow using LangGraph with interrupt support.

This workflow supports human-in-the-loop program selection using
LangGraph's native interrupt mechanism.
"""

import logging
from typing import Any, Callable, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ...models.schemas import DocumentGroup, WorkflowStatus
from ..state import WorkflowState
from ..nodes.contract_uploader import ContractUploaderNode
from ..nodes.base_info_extractor import BaseInfoExtractorNode
from ..nodes.asset_selector import AssetSelectorNode
from ..nodes.finalize import FinalizeWorkflowNode

logger = logging.getLogger(__name__)

# Global checkpointer instance for state persistence across interrupts
# In production, replace with PostgresSaver or RedisSaver for durability
_checkpointer = MemorySaver()

# Cached compiled workflow
_compiled_workflow = None


def get_workflow() -> Any:
    """
    Get or create the compiled workflow with checkpointer.

    The workflow is compiled once and cached for reuse.
    The checkpointer enables interrupt/resume functionality.
    """
    global _compiled_workflow

    if _compiled_workflow is None:
        workflow = StateGraph(WorkflowState)

        # Add nodes - instantiate classes here (callable via __call__)
        workflow.add_node("upload_documents", ContractUploaderNode())
        workflow.add_node("base_info_extractor_agent", BaseInfoExtractorNode())
        workflow.add_node("asset_selector", AssetSelectorNode())
        workflow.add_node("finalize", FinalizeWorkflowNode())

        # Set entry point
        workflow.set_entry_point("upload_documents")

        # Connect edges
        workflow.add_edge("upload_documents", "base_info_extractor_agent")
        workflow.add_edge("base_info_extractor_agent", "asset_selector")
        workflow.add_edge("asset_selector", "finalize")
        workflow.add_edge("finalize", END)

        # Compile with checkpointer for interrupt support
        _compiled_workflow = workflow.compile(checkpointer=_checkpointer)

    return _compiled_workflow


def orchestrate_blueprint_refinement_workflow(
    additional_nodes: list[tuple[str, Callable]] = None,
) -> Any:
    """
    Create the blueprint refinement workflow graph.

    DEPRECATED: Use get_workflow() instead for interrupt support.
    This function is kept for backwards compatibility.

    The workflow includes:
    1. upload_documents - Upload files to S3
    2. base_info_extractor_agent - Extract basic contract metadata using LLM
    3. asset_selector - Transform extracted info and await user program selection
    4. finalize - Complete the workflow after selection

    Args:
        additional_nodes: List of (node_name, node_function) tuples to add after finalize

    Returns:
        Compiled StateGraph workflow
    """
    # Create the graph with our state schema
    workflow = StateGraph(WorkflowState)

    # Add the core nodes - instantiate classes here (callable via __call__)
    workflow.add_node("upload_documents", ContractUploaderNode())
    workflow.add_node("base_info_extractor_agent", BaseInfoExtractorNode())
    workflow.add_node("asset_selector", AssetSelectorNode())
    workflow.add_node("finalize", FinalizeWorkflowNode())

    # Set the entry point
    workflow.set_entry_point("upload_documents")

    # Connect upload to base info extraction
    workflow.add_edge("upload_documents", "base_info_extractor_agent")

    # Connect base info extraction to program selector
    workflow.add_edge("base_info_extractor_agent", "asset_selector")

    # Connect program selector to finalize
    workflow.add_edge("asset_selector", "finalize")

    # Add any additional nodes after finalize
    if additional_nodes:
        prev_node = "finalize"
        for node_name, node_func in additional_nodes:
            workflow.add_node(node_name, node_func)
            workflow.add_edge(prev_node, node_name)
            prev_node = node_name
        # Connect last node to END
        workflow.add_edge(prev_node, END)
    else:
        # If no additional nodes, connect finalize directly to END
        workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=_checkpointer)


async def run_blueprint_refinement_workflow(
    document_group: DocumentGroup,
) -> WorkflowState:
    """
    Run the blueprint refinement workflow for a document group.

    The workflow will run until completion, failure, or an interrupt.
    When the workflow hits an interrupt (at asset_selector),
    it returns immediately with the current state. The state is
    automatically persisted by the checkpointer.

    Use resume_workflow_with_selection() to continue after interrupt.

    Args:
        document_group: The document group to process

    Returns:
        Current workflow state (may be interrupted, completed, or failed)
    """
    group_id = document_group.group_id
    logger.info(f"[{group_id}] Starting blueprint workflow")

    # Initialize the workflow state
    initial_state: WorkflowState = {
        "document_group": document_group,
        "status": WorkflowStatus.IN_PROGRESS,
        "current_step": "starting",
        "error_message": None,
        "uploaded_files": [],
        "upload_path": None,
        "page_images": None,
        "extracted_base_info": None,
        "program_selection_options": None,
        "awaiting_program_selection": False,
        "selected_program": None,
    }

    # Get the workflow with checkpointer
    workflow = get_workflow()

    # Use group_id as thread_id for state persistence
    config = {"configurable": {"thread_id": group_id}}

    # Run workflow - will return when complete, failed, or interrupted
    final_state = await workflow.ainvoke(initial_state, config)

    logger.info(
        f"[{group_id}] Blueprint workflow returned with status: {final_state.get('status')}"
    )
    return final_state


async def resume_workflow_with_selection(
    group_id: str,
    selected_program: dict[str, Any],
) -> WorkflowState:
    """
    Resume a paused workflow with the user's program selection.

    This function is called when the user selects a program from the
    selection UI. It resumes the workflow from the interrupt point
    by passing the selection to the interrupt() call.

    Args:
        group_id: The workflow thread ID (same as document group_id)
        selected_program: The user's selection containing:
            - program_name: Selected program name
            - contract_type: Contract type (optional)
            - contract_name: Contract name (optional)
            - parties: List of parties (optional)
            - date_effective: Effective date (optional)
            - date_executed: Executed date (optional)

    Returns:
        Final workflow state after completion
    """
    logger.info(
        f"[{group_id}] Resuming workflow with selection: {selected_program.get('program_name')}"
    )

    workflow = get_workflow()
    config = {"configurable": {"thread_id": group_id}}

    # Update state to set the awaiting flag BEFORE resuming
    # This allows the node to detect it's resuming and skip duplicate progress updates
    workflow.update_state(
        config,
        {"awaiting_program_selection": True},
    )

    # Resume the workflow by passing the selection to the interrupt
    # The Command(resume=...) sends the value back to interrupt()
    final_state = await workflow.ainvoke(
        Command(resume=selected_program),
        config,
    )

    logger.info(
        f"[{group_id}] Workflow completed with status: {final_state.get('status')}"
    )
    return final_state


def get_workflow_snapshot(group_id: str) -> Optional[Any]:
    """
    Get the current state snapshot for a workflow.

    Useful for debugging or checking if a workflow is interrupted.

    Args:
        group_id: The workflow thread ID

    Returns:
        The current state snapshot or None if not found
    """
    workflow = get_workflow()
    config = {"configurable": {"thread_id": group_id}}

    try:
        return workflow.get_state(config)
    except Exception as e:
        logger.warning(f"[{group_id}] Failed to get workflow snapshot: {e}")
        return None
