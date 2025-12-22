"""
Unit tests for Contract Processing API endpoints
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.schemas import (
    DocumentType,
    FileType,
    WorkflowStatus,
    DocumentMetadata,
    DocumentGroup,
)
from app.services.contract_service import ContractService


# Sync test client for simple tests
client = TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns application info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["application"] == "Contract Assistance API"
        assert data["status"] == "running"

    def test_health_endpoint(self):
        """Test global health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_contracts_health_endpoint(self):
        """Test contracts router health endpoint"""
        response = client.get("/api/contracts/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "contract-assistance-api"


class TestProcessContractInference:
    """Tests for processContractInference endpoint"""

    def setup_method(self):
        """Clear workflow store before each test"""
        ContractService._workflow_store.clear()
        ContractService._callbacks.clear()

    def test_process_contract_inference_standalone(self):
        """Test processing a standalone contract"""
        document_group = {
            "group_id": "test-standalone-001",
            "identifier_name": "Test Standalone Contract",
            "document_type": "standalone",
            "documents": [
                {
                    "filename": "contract.pdf",
                    "file_type": "STANDALONE",
                    "size_bytes": 1024,
                    "content_type": "application/pdf",
                }
            ],
            "refine_blueprints": False,
        }

        response = client.post(
            "/api/contracts/processContractInference",
            json=document_group,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["group_id"] == "test-standalone-001"
        assert data["status"] == "in_progress"
        assert "processing started" in data["message"].lower()

    def test_process_contract_inference_master_with_attachments(self):
        """Test processing a master contract with attachments"""
        document_group = {
            "group_id": "test-master-001",
            "identifier_name": "Master Contract with Exhibits",
            "document_type": "master",
            "documents": [
                {
                    "filename": "master_agreement.pdf",
                    "file_type": "MASTER",
                    "size_bytes": 2048,
                    "content_type": "application/pdf",
                },
                {
                    "filename": "exhibit_a.pdf",
                    "file_type": "ATTACHMENT",
                    "size_bytes": 512,
                    "content_type": "application/pdf",
                },
                {
                    "filename": "exhibit_b.pdf",
                    "file_type": "ATTACHMENT",
                    "size_bytes": 768,
                    "content_type": "application/pdf",
                },
            ],
            "refine_blueprints": False,
        }

        response = client.post(
            "/api/contracts/processContractInference",
            json=document_group,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["group_id"] == "test-master-001"
        assert data["status"] == "in_progress"

    def test_process_contract_inference_waiver(self):
        """Test processing a waiver contract"""
        document_group = {
            "group_id": "test-waiver-001",
            "identifier_name": "Fee Waiver Document",
            "document_type": "waiver",
            "documents": [
                {
                    "filename": "waiver.pdf",
                    "file_type": "WAIVER",
                    "size_bytes": 256,
                    "content_type": "application/pdf",
                }
            ],
            "refine_blueprints": False,
        }

        response = client.post(
            "/api/contracts/processContractInference",
            json=document_group,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["group_id"] == "test-waiver-001"
        assert data["status"] == "in_progress"

    def test_process_contract_inference_invalid_payload(self):
        """Test processing with invalid payload"""
        invalid_payload = {
            "group_id": "test-invalid-001",
            # Missing required fields
        }

        response = client.post(
            "/api/contracts/processContractInference",
            json=invalid_payload,
        )

        assert response.status_code == 422  # Validation error


class TestProcessBlueprintsRefinement:
    """Tests for processBlueprintsRefinement endpoint"""

    def setup_method(self):
        """Clear workflow store before each test"""
        ContractService._workflow_store.clear()
        ContractService._callbacks.clear()

    def test_process_blueprints_refinement(self):
        """Test processing with blueprint refinement enabled"""
        document_group = {
            "group_id": "test-blueprint-001",
            "identifier_name": "Contract with Blueprint Refinement",
            "document_type": "standalone",
            "documents": [
                {
                    "filename": "contract.pdf",
                    "file_type": "STANDALONE",
                    "size_bytes": 1024,
                    "content_type": "application/pdf",
                }
            ],
            "refine_blueprints": True,
        }

        response = client.post(
            "/api/contracts/processBlueprintsRefinement",
            json=document_group,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["group_id"] == "test-blueprint-001"
        assert data["status"] == "in_progress"
        assert "blueprint refinement" in data["message"].lower()

    def test_process_blueprints_refinement_master(self):
        """Test blueprint refinement with master contract"""
        document_group = {
            "group_id": "test-blueprint-master-001",
            "identifier_name": "Master with Blueprints",
            "document_type": "master",
            "documents": [
                {
                    "filename": "master.pdf",
                    "file_type": "MASTER",
                    "size_bytes": 2048,
                },
                {
                    "filename": "attachment.pdf",
                    "file_type": "ATTACHMENT",
                    "size_bytes": 512,
                },
            ],
            "refine_blueprints": True,
        }

        response = client.post(
            "/api/contracts/processBlueprintsRefinement",
            json=document_group,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"


class TestWorkflowProgress:
    """Tests for workflow progress endpoints"""

    def setup_method(self):
        """Clear workflow store before each test"""
        ContractService._workflow_store.clear()
        ContractService._callbacks.clear()

    def test_get_progress_not_found(self):
        """Test getting progress for non-existent workflow"""
        response = client.get("/api/contracts/progress/non-existent-id")
        assert response.status_code == 404

    def test_clear_workflow_progress(self):
        """Test clearing workflow progress"""
        response = client.delete("/api/contracts/progress/test-clear-001")
        assert response.status_code == 200
        assert "cleared" in response.json()["message"].lower()


@pytest.mark.asyncio
class TestAsyncWorkflowProcessing:
    """Async tests for workflow processing"""

    async def setup_method(self):
        """Clear workflow store before each test"""
        ContractService._workflow_store.clear()
        ContractService._callbacks.clear()

    async def test_contract_inference_full_workflow(self):
        """Test full contract inference workflow execution"""
        ContractService._workflow_store.clear()
        
        document_group = DocumentGroup(
            group_id="async-test-001",
            identifier_name="Async Test Contract",
            document_type=DocumentType.STANDALONE,
            documents=[
                DocumentMetadata(
                    filename="test.pdf",
                    file_type=FileType.STANDALONE,
                    size_bytes=1024,
                )
            ],
            refine_blueprints=False,
        )

        # Run the full processing
        result = await ContractService.process_contract_inference(document_group)

        assert result.group_id == "async-test-001"
        assert result.status == WorkflowStatus.COMPLETED
        assert result.workflow_progress is not None
        assert len(result.workflow_progress.steps) > 0

    async def test_blueprint_refinement_full_workflow(self):
        """Test full blueprint refinement workflow execution"""
        ContractService._workflow_store.clear()
        
        document_group = DocumentGroup(
            group_id="async-blueprint-001",
            identifier_name="Async Blueprint Test",
            document_type=DocumentType.MASTER,
            documents=[
                DocumentMetadata(
                    filename="master.pdf",
                    file_type=FileType.MASTER,
                    size_bytes=2048,
                ),
                DocumentMetadata(
                    filename="attachment.pdf",
                    file_type=FileType.ATTACHMENT,
                    size_bytes=512,
                ),
            ],
            refine_blueprints=True,
        )

        # Run the full processing
        result = await ContractService.process_blueprints_refinement(document_group)

        assert result.group_id == "async-blueprint-001"
        assert result.status == WorkflowStatus.COMPLETED
        assert result.workflow_progress is not None
        # Blueprint refinement has more steps
        assert len(result.workflow_progress.steps) >= 4

    async def test_parallel_processing(self):
        """Test parallel processing of multiple document groups"""
        ContractService._workflow_store.clear()
        
        groups = [
            DocumentGroup(
                group_id=f"parallel-{i}",
                identifier_name=f"Parallel Contract {i}",
                document_type=DocumentType.STANDALONE,
                documents=[
                    DocumentMetadata(
                        filename=f"contract_{i}.pdf",
                        file_type=FileType.STANDALONE,
                        size_bytes=1024,
                    )
                ],
                refine_blueprints=False,
            )
            for i in range(3)
        ]

        # Process all groups in parallel
        results = await asyncio.gather(
            *[ContractService.process_contract_inference(group) for group in groups]
        )

        # All should complete successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.group_id == f"parallel-{i}"
            assert result.status == WorkflowStatus.COMPLETED


class TestDocumentModels:
    """Tests for document model validation"""

    def test_valid_document_group(self):
        """Test valid document group creation"""
        group = DocumentGroup(
            group_id="model-test-001",
            identifier_name="Model Test",
            document_type=DocumentType.STANDALONE,
            documents=[
                DocumentMetadata(
                    filename="test.pdf",
                    file_type=FileType.STANDALONE,
                    size_bytes=1024,
                )
            ],
        )

        assert group.group_id == "model-test-001"
        assert group.refine_blueprints is True  # Default value
        assert len(group.documents) == 1

    def test_document_type_enum(self):
        """Test document type enum values"""
        assert DocumentType.STANDALONE.value == "standalone"
        assert DocumentType.MASTER.value == "master"
        assert DocumentType.WAIVER.value == "waiver"

    def test_file_type_enum(self):
        """Test file type enum values"""
        assert FileType.MASTER.value == "MASTER"
        assert FileType.ATTACHMENT.value == "ATTACHMENT"
        assert FileType.STANDALONE.value == "STANDALONE"
        assert FileType.WAIVER.value == "WAIVER"

    def test_workflow_status_enum(self):
        """Test workflow status enum values"""
        assert WorkflowStatus.IN_PROGRESS.value == "in_progress"
        assert WorkflowStatus.AWAITING_FEEDBACK.value == "awaiting_feedback"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"

