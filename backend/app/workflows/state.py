"""
Workflow state definitions for LangGraph workflows
"""

from typing import Optional, TypedDict

from ..models.schemas import DocumentGroup, WorkflowStatus


class WorkflowState(TypedDict):
    """
    State object that flows through the LangGraph workflow.
    Each node can read and update this state.
    """

    # Input data
    document_group: DocumentGroup

    # Processing status
    status: WorkflowStatus
    current_step: str
    error_message: Optional[str]

    # Upload results
    uploaded_files: list[str]  # S3 keys of uploaded files
    upload_path: Optional[str]  # Base S3 path where files are uploaded

    # Page images mapping: {file_type: {page_number: s3_uri}}
    page_images: Optional[dict[str, dict[int, str]]]

