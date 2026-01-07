"""
Base workflow node class with common functionality.

All workflow nodes should extend BaseWorkflowNode to get:
- Standardized progress updates
- Common context extraction from state
- Consistent error/success response creation
- LangGraph compatibility via __call__
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from models.schemas import WorkflowStatus
from workflows.state import WorkflowState


class BaseWorkflowNode(ABC):
    """
    Abstract base class for all workflow nodes.

    Provides common functionality:
    - Progress update helper that communicates with ContractService
    - Context extraction from workflow state
    - Standardized error and success response builders
    - Callable interface for LangGraph compatibility

    Usage:
        class MyNode(BaseWorkflowNode):
            def __init__(self):
                super().__init__("MyNode")

            async def execute(self, state: WorkflowState) -> dict[str, Any]:
                group_id, identifier_name, doc_group = self._get_context(state)
                # ... node logic ...
                return self._create_success_response(step_id="my_step", ...)

        # Create instance for LangGraph
        my_node = MyNode()
    """

    def __init__(self, node_name: str):
        """
        Initialize the node with a name for logging.

        Args:
            node_name: Human-readable name for this node (used in logs)
        """
        self.node_name = node_name
        self.logger = logging.getLogger(f"{__name__}.{node_name}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Context Extraction
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_context(self, state: WorkflowState) -> tuple[str, str, Any]:
        """
        Extract common context from workflow state.

        Args:
            state: Current workflow state

        Returns:
            Tuple of (group_id, identifier_name, document_group)
        """
        document_group = state["document_group"]
        return document_group.group_id, document_group.identifier_name, document_group

    def _is_already_failed(self, state: WorkflowState) -> bool:
        """
        Check if workflow has already failed (skip processing).

        Args:
            state: Current workflow state

        Returns:
            True if status is FAILED, False otherwise
        """
        return state.get("status") == WorkflowStatus.FAILED

    # ═══════════════════════════════════════════════════════════════════════════
    # Progress Updates
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_progress(
        self,
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

        Uses late import to avoid circular dependency with ContractService.

        Args:
            group_id: Unique identifier for the document group
            identifier_name: User-defined name for the document group
            step_id: Unique identifier for this step
            step_name: Human-readable name for this step
            status: Current workflow status
            message: Progress message to display
            requires_feedback: Whether this step requires user input
            selection_data: Optional selection options for user feedback
        """
        try:
            from services.contract_service import ContractService

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
            self.logger.warning(f"Failed to update workflow progress: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Response Builders
    # ═══════════════════════════════════════════════════════════════════════════

    def _create_error_response(
        self,
        step_id: str,
        error_message: str,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """
        Create a standardized error response dictionary.

        Args:
            step_id: The step where the error occurred
            error_message: Description of the error
            **extra_fields: Additional fields to include in the response

        Returns:
            Dictionary with status=FAILED and error details
        """
        return {
            "status": WorkflowStatus.FAILED,
            "current_step": step_id,
            "error_message": error_message,
            **extra_fields,
        }

    def _create_success_response(
        self,
        step_id: str,
        status: WorkflowStatus = WorkflowStatus.IN_PROGRESS,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """
        Create a standardized success response dictionary.

        Args:
            step_id: The completed step identifier
            status: Workflow status (defaults to IN_PROGRESS)
            **extra_fields: Additional fields to include in the response

        Returns:
            Dictionary with success status and provided fields
        """
        return {
            "status": status,
            "current_step": step_id,
            "error_message": None,
            **extra_fields,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Abstract Method
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    async def execute(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the node logic.

        Must be implemented by all subclasses.

        Args:
            state: Current workflow state

        Returns:
            Dictionary of state updates
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # LangGraph Compatibility
    # ═══════════════════════════════════════════════════════════════════════════

    async def __call__(self, state: WorkflowState) -> dict[str, Any]:
        """
        Make node instances callable for LangGraph.

        This allows node instances to be used directly in workflow.add_node().

        Args:
            state: Current workflow state

        Returns:
            Result of execute() method
        """
        return await self.execute(state)

