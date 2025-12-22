"""
Blueprint refinement workflow using LangGraph
"""

import logging
from typing import Callable

from langgraph.graph import StateGraph, END

from ...models.schemas import DocumentGroup, WorkflowStatus
from ..state import WorkflowState
from ..nodes.upload_node import upload_documents

logger = logging.getLogger(__name__)


def orchestrate_blueprint_refinement_workflow(
    additional_nodes: list[tuple[str, Callable]] = None,
) -> StateGraph:
    """
    Create the blueprint refinement workflow graph.

    The workflow currently includes:
    1. upload_documents - Upload files to S3

    Additional nodes can be added via the additional_nodes parameter.

    Args:
        additional_nodes: List of (node_name, node_function) tuples to add after upload

    Returns:
        Compiled StateGraph workflow
    """
    # Create the graph with our state schema
    workflow = StateGraph(WorkflowState)

    # Add the upload node
    workflow.add_node("upload_documents", upload_documents)

    # Set the entry point
    workflow.set_entry_point("upload_documents")

    # Add any additional nodes
    if additional_nodes:
        prev_node = "upload_documents"
        for node_name, node_func in additional_nodes:
            workflow.add_node(node_name, node_func)
            workflow.add_edge(prev_node, node_name)
            prev_node = node_name
        # Connect last node to END
        workflow.add_edge(prev_node, END)
    else:
        # If no additional nodes, connect upload directly to END
        workflow.add_edge("upload_documents", END)

    return workflow.compile()


async def run_blueprint_refinement_workflow(
    document_group: DocumentGroup,
) -> WorkflowState:
    """
    Run the blueprint refinement workflow for a document group.

    Args:
        document_group: The document group to process

    Returns:
        Final workflow state after completion
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
    }

    # Create and run the workflow
    workflow = orchestrate_blueprint_refinement_workflow()
    final_state = await workflow.ainvoke(initial_state)

    logger.info(
        f"[{group_id}] Blueprint workflow completed with status: {final_state['status']}"
    )
    return final_state
