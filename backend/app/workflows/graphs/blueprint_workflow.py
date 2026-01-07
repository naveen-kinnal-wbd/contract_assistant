"""
Blueprint refinement workflow using LangGraph
"""

import logging
from typing import Callable

from langgraph.graph import StateGraph, END

from ...models.schemas import DocumentGroup, WorkflowStatus
from ..state import WorkflowState
from ..nodes.upload_node import upload_documents
from ..nodes.base_info_extractor_node import base_info_extractor_agent
from ..nodes.program_selector_node import program_selector_node

logger = logging.getLogger(__name__)


def orchestrate_blueprint_refinement_workflow(
    additional_nodes: list[tuple[str, Callable]] = None,
) -> StateGraph:
    """
    Create the blueprint refinement workflow graph.

    The workflow includes:
    1. upload_documents - Upload files to S3
    2. base_info_extractor_agent - Extract basic contract metadata using LLM
    3. program_selector_node - Transform extracted info and await user program selection

    Additional nodes can be added via the additional_nodes parameter.

    Args:
        additional_nodes: List of (node_name, node_function) tuples to add after program selection

    Returns:
        Compiled StateGraph workflow
    """
    # Create the graph with our state schema
    workflow = StateGraph(WorkflowState)

    # Add the core nodes
    workflow.add_node("upload_documents", upload_documents)
    workflow.add_node("base_info_extractor_agent", base_info_extractor_agent)
    workflow.add_node("program_selector_node", program_selector_node)

    # Set the entry point
    workflow.set_entry_point("upload_documents")

    # Connect upload to base info extraction
    workflow.add_edge("upload_documents", "base_info_extractor_agent")

    # Connect base info extraction to program selector
    workflow.add_edge("base_info_extractor_agent", "program_selector_node")

    # Add any additional nodes after program_selector_node
    if additional_nodes:
        prev_node = "program_selector_node"
        for node_name, node_func in additional_nodes:
            workflow.add_node(node_name, node_func)
            workflow.add_edge(prev_node, node_name)
            prev_node = node_name
        # Connect last node to END
        workflow.add_edge(prev_node, END)
    else:
        # If no additional nodes, connect program selector directly to END
        workflow.add_edge("program_selector_node", END)

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
        "page_images": None,
        "extracted_base_info": None,
        "program_selection_options": None,
        "selected_program": None,
    }

    # Create and run the workflow
    workflow = orchestrate_blueprint_refinement_workflow()
    final_state = await workflow.ainvoke(initial_state)

    logger.info(
        f"[{group_id}] Blueprint workflow completed with status: {final_state['status']}"
    )
    return final_state
