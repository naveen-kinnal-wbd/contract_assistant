"""
Program selector node for LangGraph workflow with interrupt support.

This node transforms extracted base info into program selection options
and uses LangGraph's interrupt() to pause for user selection.
"""

import logging
from typing import Any

from langgraph.types import interrupt

from ...models.schemas import WorkflowStatus
from ..state import WorkflowState

logger = logging.getLogger(__name__)


def _update_workflow_progress(
    group_id: str,
    identifier_name: str,
    step_id: str,
    step_name: str,
    status: WorkflowStatus,
    message: str,
    requires_feedback: bool = False,
    selection_data: list[dict] | None = None,
) -> None:
    """
    Update workflow progress in the ContractService store.
    Uses late import to avoid circular dependency.
    """
    try:
        from ...services.contract_service import ContractService

        ContractService._update_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id=step_id,
            step_name=step_name,
            status=status,
            message=message,
            requires_feedback=requires_feedback,
            selection_data=selection_data,
        )
    except Exception as e:
        logger.warning(f"Failed to update workflow progress: {e}")


def transform_to_program_options(
    extracted_base_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Transform extracted base info into a list of program selection options.

    Each program gets combined with the shared contract metadata to create
    a complete row for the selection table.

    Args:
        extracted_base_info: The extracted contract metadata from base_info_extractor

    Returns:
        List of program option dictionaries suitable for table display
    """
    if not extracted_base_info:
        return []

    # Extract shared fields (these apply to all programs)
    contract_type_data = extracted_base_info.get("contract_type")
    contract_type = None
    if contract_type_data:
        if isinstance(contract_type_data, dict):
            contract_type = contract_type_data.get("value")
        elif isinstance(contract_type_data, str):
            contract_type = contract_type_data

    contract_name_data = extracted_base_info.get("contract_name")
    contract_name = None
    if contract_name_data:
        if isinstance(contract_name_data, dict):
            contract_name = contract_name_data.get("value")
        elif isinstance(contract_name_data, str):
            contract_name = contract_name_data

    date_effective_data = extracted_base_info.get("date_effective")
    date_effective = None
    if date_effective_data:
        if isinstance(date_effective_data, dict):
            date_effective = date_effective_data.get("value")
        elif isinstance(date_effective_data, str):
            date_effective = date_effective_data

    date_executed_data = extracted_base_info.get("date_executed")
    date_executed = None
    if date_executed_data:
        if isinstance(date_executed_data, dict):
            date_executed = date_executed_data.get("value")
        elif isinstance(date_executed_data, str):
            date_executed = date_executed_data

    # Extract parties (list field)
    parties_data = extracted_base_info.get("parties", [])
    parties = []
    if isinstance(parties_data, list):
        for party in parties_data:
            if isinstance(party, dict):
                party_value = party.get("value")
                if party_value:
                    parties.append(party_value)
            elif isinstance(party, str):
                parties.append(party)

    # Extract programs (list field) - each becomes a selection option
    programs_data = extracted_base_info.get("programs", [])
    program_options = []

    if isinstance(programs_data, list):
        for program in programs_data:
            program_name = None
            if isinstance(program, dict):
                program_name = program.get("value")
            elif isinstance(program, str):
                program_name = program

            if program_name:
                program_options.append(
                    {
                        "program_name": program_name,
                        "contract_type": contract_type,
                        "contract_name": contract_name,
                        "parties": parties,
                        "date_effective": date_effective,
                        "date_executed": date_executed,
                    }
                )

    # If no programs found, create a single option with just the contract info
    if not program_options and (contract_type or contract_name):
        program_options.append(
            {
                "program_name": "N/A",
                "contract_type": contract_type,
                "contract_name": contract_name,
                "parties": parties,
                "date_effective": date_effective,
                "date_executed": date_executed,
            }
        )

    return program_options


async def asset_selector(state: WorkflowState) -> dict[str, Any]:
    """
    Transform extracted base info into program selection options and
    use interrupt() to pause for user selection.

    Flow:
    1. Reads extracted_base_info from state
    2. Transforms programs into selectable options with all metadata
    3. Updates workflow progress to AWAITING_FEEDBACK with selection_data
    4. Calls interrupt() - workflow pauses here
    5. When resumed, interrupt() returns the selected program
    6. Updates progress and returns state with selected_program

    Resume Detection:
    When the UI triggers a resume, it first updates the state with
    awaiting_program_selection=True via workflow.update_state().
    This allows the node to detect it's resuming and skip duplicate
    progress updates.

    Args:
        state: Current workflow state containing extracted_base_info

    Returns:
        Updated state fields with selected_program after user selection
    """
    document_group = state["document_group"]
    group_id = document_group.group_id
    identifier_name = document_group.identifier_name
    extracted_base_info = state.get("extracted_base_info")

    # Check if we're resuming from interrupt using state flag
    # The flag is set by resume_workflow_with_selection() before resuming
    # This is a clean, decoupled way to detect resume without checking progress store
    is_resuming = state.get("awaiting_program_selection", False)

    if is_resuming:
        logger.info(f"[{group_id}] Resuming program selection from interrupt")
    else:
        logger.info(f"[{group_id}] Starting program selection")

    # Check for prior failure
    if state.get("status") == WorkflowStatus.FAILED:
        logger.warning(
            f"[{group_id}] Skipping program selector - workflow already failed"
        )
        return {}

    # Check if we have extracted info
    if not extracted_base_info:
        error_msg = "No extracted base info available for program selection"
        logger.error(f"[{group_id}] {error_msg}")

        _update_workflow_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="program_selection_failed",
            step_name="Program Selection",
            status=WorkflowStatus.FAILED,
            message=error_msg,
        )

        return {
            "status": WorkflowStatus.FAILED,
            "current_step": "program_selection_failed",
            "error_message": error_msg,
            "program_selection_options": None,
        }

    # Transform extracted info into program options
    # This is wrapped in try/except for actual errors
    try:
        program_options = transform_to_program_options(extracted_base_info)
    except Exception as e:
        error_msg = f"Failed to transform program options: {str(e)}"
        logger.error(f"[{group_id}] {error_msg}")

        _update_workflow_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="program_selection_failed",
            step_name="Program Selection",
            status=WorkflowStatus.FAILED,
            message=error_msg,
        )

        return {
            "status": WorkflowStatus.FAILED,
            "current_step": "program_selection_failed",
            "error_message": error_msg,
            "program_selection_options": None,
        }

    if not program_options:
        error_msg = "No programs found in extracted contract information"
        logger.warning(f"[{group_id}] {error_msg}")

        _update_workflow_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="program_selection_failed",
            step_name="Program Selection",
            status=WorkflowStatus.FAILED,
            message=error_msg,
        )

        return {
            "status": WorkflowStatus.FAILED,
            "current_step": "program_selection_failed",
            "error_message": error_msg,
            "program_selection_options": None,
        }

    logger.info(f"[{group_id}] Found {len(program_options)} program(s) for selection")

    # Only update progress to AWAITING_FEEDBACK on first run, not on resume
    # (On resume, the node re-executes but we don't want duplicate steps)
    if not is_resuming:
        _update_workflow_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="program_selection",
            step_name="Program Selection",
            status=WorkflowStatus.AWAITING_FEEDBACK,
            message="The following programs have been extracted from the contract. Please select a program to continue.",
            requires_feedback=True,
            selection_data=program_options,
        )
        logger.info(f"[{group_id}] Awaiting user program selection (interrupt)")

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERRUPT: Workflow pauses here and waits for human input.
    # When resumed with Command(resume=selected_program), interrupt()
    # returns that value and execution continues below.
    #
    # IMPORTANT: Do NOT wrap interrupt() in try/except - LangGraph uses a special
    # exception mechanism to pause execution, and catching it will break the flow.
    # ═══════════════════════════════════════════════════════════════════════════
    selected_program = interrupt(
        {
            "type": "program_selection",
            "options": program_options,
            "message": "Select a program to continue processing",
        }
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # RESUMED: Continue processing with user's selection
    # ═══════════════════════════════════════════════════════════════════════════
    program_name = selected_program.get("program_name", "Unknown")
    logger.info(f"[{group_id}] Program selected: {program_name}")

    # Update progress - selection received
    # This marks the AWAITING_FEEDBACK step as completed and shows the selection
    _update_workflow_progress(
        group_id=group_id,
        identifier_name=identifier_name,
        step_id="program_selected",
        step_name="Program Selected",
        status=WorkflowStatus.COMPLETED,
        message=f"Program selected: {program_name}",
    )

    return {
        "status": WorkflowStatus.IN_PROGRESS,
        "current_step": "program_selected",
        "error_message": None,
        "program_selection_options": program_options,
        "selected_program": selected_program,
        # Flag indicating program selection is complete (useful for downstream logic)
        "awaiting_program_selection": False,
    }
