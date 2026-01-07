"""
Pydantic schemas for Contract Assistance API
"""

from enum import Enum
from typing import Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Type of document group being uploaded"""

    STANDALONE = "standalone"
    MASTER = "master"
    WAIVER = "waiver"


class FileType(str, Enum):
    """Type classification for individual files"""

    MASTER = "MASTER"
    ATTACHMENT = "ATTACHMENT"
    STANDALONE = "STANDALONE"
    WAIVER = "WAIVER"


class WorkflowStatus(str, Enum):
    """Status of the document processing workflow"""

    IN_PROGRESS = "in_progress"
    AWAITING_FEEDBACK = "awaiting_feedback"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata for an individual document file"""

    filename: str
    file_type: FileType
    size_bytes: int = 0
    content_type: Optional[str] = None
    content: Optional[str] = None  # Base64-encoded file content


class DocumentGroup(BaseModel):
    """Request model for document group processing"""

    group_id: str = Field(..., description="Unique identifier for the document group")
    identifier_name: str = Field(
        ..., description="User-defined nickname for the document group"
    )
    document_type: DocumentType
    documents: list[DocumentMetadata]
    refine_blueprints: bool = True


class WorkflowStep(BaseModel):
    """Individual step in the processing workflow"""

    step_id: str
    step_name: str
    status: WorkflowStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    requires_feedback: bool = False
    selection_data: Optional[list[dict]] = None  # For asset selection options


class WorkflowProgress(BaseModel):
    """Progress update for document processing workflow"""

    group_id: str
    identifier_name: str
    current_status: WorkflowStatus
    steps: list[WorkflowStep] = []
    current_step_index: int = 0


class ProcessingResponse(BaseModel):
    """Response model for processing endpoints"""

    group_id: str
    status: WorkflowStatus
    message: str
    workflow_progress: Optional[WorkflowProgress] = None


class AssetSelectionRequest(BaseModel):
    """Request model for asset selection"""

    group_id: str
    deal_id: str
    asset_id: str
    deal_name: str
    asset_name: str


class ProgramSelectionRequest(BaseModel):
    """Request model for program selection"""

    group_id: str
    program_name: str
    contract_type: Optional[str] = None
    contract_name: Optional[str] = None
    parties: Optional[list[Any]] = None  # Can be list of strings or dicts
    date_effective: Optional[str] = None
    date_executed: Optional[str] = None
