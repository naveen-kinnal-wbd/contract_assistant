"""
Workflow nodes for LangGraph workflows.

This module provides workflow node classes that extend BaseWorkflowNode.

Usage with LangGraph:
    from ..nodes import ContractUploaderNode, BaseInfoExtractorNode, AssetSelectorNode

    workflow.add_node("upload_documents", ContractUploaderNode())
    workflow.add_node("base_info_extractor_agent", BaseInfoExtractorNode())
    workflow.add_node("asset_selector", AssetSelectorNode())

Direct class usage (for testing):
    from ..nodes import ContractUploaderNode

    node = ContractUploaderNode()
    result = await node.execute(state)
"""

# Base class for all workflow nodes
from workflows.nodes.base import BaseWorkflowNode

# Node classes
from workflows.nodes.contract_uploader import ContractUploaderNode
from workflows.nodes.base_info_extractor import BaseInfoExtractorNode
from workflows.nodes.asset_selector import AssetSelectorNode
from workflows.nodes.media_rights_extractor import MediaRightsExtractorNode
from workflows.nodes.finalize import FinalizeWorkflowNode

# Helper functions that may be useful externally
from workflows.nodes.base_info_extractor import (
    parse_llm_response,
    aggregate_page_extractions,
    fetch_image_from_s3,
    parse_s3_uri,
    BASE_INFO_SCHEMA,
)

__all__ = [
    # Base class
    "BaseWorkflowNode",
    # Node classes
    "ContractUploaderNode",
    "BaseInfoExtractorNode",
    "AssetSelectorNode",
    "MediaRightsExtractorNode",
    "FinalizeWorkflowNode",
    # Helper functions
    "parse_llm_response",
    "aggregate_page_extractions",
    "fetch_image_from_s3",
    "parse_s3_uri",
    "BASE_INFO_SCHEMA",
]
