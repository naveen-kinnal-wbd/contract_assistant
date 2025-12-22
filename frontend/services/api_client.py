"""
API client for communicating with Contract Assistance backend
"""

import os
import httpx
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class WorkflowStatus(str, Enum):
    """Status of the document processing workflow"""

    IN_PROGRESS = "in_progress"
    AWAITING_FEEDBACK = "awaiting_feedback"
    COMPLETED = "completed"
    FAILED = "failed"


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


@dataclass
class DocumentMetadata:
    """Metadata for an individual document file"""

    filename: str
    file_type: FileType
    size_bytes: int = 0
    content_type: Optional[str] = None

    def to_dict(self):
        return {
            "filename": self.filename,
            "file_type": self.file_type.value,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
        }


@dataclass
class DocumentGroup:
    """Document group for processing"""

    group_id: str
    identifier_name: str
    document_type: DocumentType
    documents: list[DocumentMetadata]
    refine_blueprints: bool = True

    def to_dict(self):
        return {
            "group_id": self.group_id,
            "identifier_name": self.identifier_name,
            "document_type": self.document_type.value,
            "documents": [doc.to_dict() for doc in self.documents],
            "refine_blueprints": self.refine_blueprints,
        }


class ContractAPIClient:
    """Client for interacting with Contract Assistance API"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.timeout = httpx.Timeout(10.0, connect=5.0)

    def _get_client(self) -> httpx.Client:
        """Create a new HTTP client"""
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def health_check(self) -> bool:
        """Check if the API is healthy"""
        try:
            with self._get_client() as client:
                response = client.get("/health")
                return response.status_code == 200
        except Exception:
            return False

    def process_contract_inference(self, document_group: DocumentGroup) -> dict:
        """
        Start contract inference processing (without blueprint refinement)
        """
        with self._get_client() as client:
            response = client.post(
                "/api/contracts/processContractInference",
                json=document_group.to_dict(),
            )
            response.raise_for_status()
            return response.json()

    def process_blueprints_refinement(self, document_group: DocumentGroup) -> dict:
        """
        Start blueprint refinement processing
        """
        with self._get_client() as client:
            response = client.post(
                "/api/contracts/processBlueprintsRefinement",
                json=document_group.to_dict(),
            )
            response.raise_for_status()
            return response.json()

    def get_workflow_progress(self, group_id: str) -> Optional[dict]:
        """
        Get current workflow progress for a document group
        """
        try:
            with self._get_client() as client:
                response = client.get(f"/api/contracts/progress/{group_id}")
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def clear_workflow_progress(self, group_id: str) -> bool:
        """
        Clear workflow progress for a document group
        """
        try:
            with self._get_client() as client:
                response = client.delete(f"/api/contracts/progress/{group_id}")
                return response.status_code == 200
        except Exception:
            return False

    def select_asset(self, group_id: str, selection: dict) -> dict:
        """
        Submit asset selection to continue the workflow

        Args:
            group_id: The workflow group ID
            selection: Dict with keys deal_id, asset_id, deal_name, asset_name
        """
        with self._get_client() as client:
            response = client.post(
                "/api/contracts/select-asset",
                json={"group_id": group_id, **selection},
            )
            response.raise_for_status()
            return response.json()
