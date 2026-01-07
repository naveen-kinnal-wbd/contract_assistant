"""
Finalize workflow node for LangGraph workflow.

This node completes the workflow after program selection,
marking the workflow as completed with final details.
"""

import logging
from typing import Any

from models.schemas import WorkflowStatus
from workflows.state import WorkflowState
from workflows.nodes.base import BaseWorkflowNode

logger = logging.getLogger(__name__)


class FinalizeWorkflowNode(BaseWorkflowNode):
    """
    Final node to complete the workflow after program selection.

    This node runs after the user has selected a program and
    marks the workflow as completed with summary details.
    """

    def __init__(self):
        super().__init__("FinalizeWorkflow")

    async def execute(self, state: WorkflowState) -> dict[str, Any]:
        """
        Finalize the workflow with completion status.

        Args:
            state: Current workflow state with selected_program

        Returns:
            Updated state with COMPLETED status
        """
        group_id, identifier_name, document_group = self._get_context(state)
        selected_program = state.get("selected_program")

        # Skip if workflow already failed
        if self._is_already_failed(state):
            return {}

        self.logger.info(f"[{group_id}] Finalizing workflow")

        program_name = "Unknown"
        if selected_program:
            program_name = selected_program.get("program_name", "Unknown")

        # Build completion message with selected program details
        details_parts = [f"Program: {program_name}"]
        if selected_program:
            if selected_program.get("contract_type"):
                details_parts.append(
                    f"Contract Type: {selected_program['contract_type']}"
                )
            if selected_program.get("contract_name"):
                details_parts.append(
                    f"Contract Name: {selected_program['contract_name']}"
                )
            if selected_program.get("date_effective"):
                details_parts.append(
                    f"Effective Date: {selected_program['date_effective']}"
                )
            if selected_program.get("date_executed"):
                details_parts.append(
                    f"Executed Date: {selected_program['date_executed']}"
                )
            if selected_program.get("parties"):
                party_names = [
                    p.get("value", str(p)) if isinstance(p, dict) else str(p)
                    for p in selected_program["parties"]
                ]
                if party_names:
                    details_parts.append(f"Parties: {', '.join(party_names)}")

        details_message = " | ".join(details_parts)

        self._update_progress(
            group_id=group_id,
            identifier_name=identifier_name,
            step_id="complete",
            step_name="Processing Complete",
            status=WorkflowStatus.COMPLETED,
            message=f"Blueprint refinement completed. {details_message}",
        )

        self.logger.info(f"[{group_id}] Workflow completed for program: {program_name}")

        return self._create_success_response(
            step_id="complete",
            status=WorkflowStatus.COMPLETED,
        )
