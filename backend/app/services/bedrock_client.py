"""
Generic AWS Bedrock LLM client for making requests with images.

This module provides a reusable client for interacting with AWS Bedrock,
supporting image-based prompts and parallel processing.
"""

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Optional

import boto3
from botocore.config import Config as BotoConfig

from config import get_llm_config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_bedrock_client():
    """
    Get cached Bedrock runtime client with configured region and timeouts.
    
    Returns:
        boto3 bedrock-runtime client
    """
    llm_config = get_llm_config()
    region = llm_config.get("region", "us-east-1")
    read_timeout = llm_config.get("read_timeout", 1800)
    connect_timeout = llm_config.get("connect_timeout", 300)
    max_retries = llm_config.get("max_retries", 5)

    boto_config = BotoConfig(
        region_name=region,
        retries={"max_attempts": max_retries, "mode": "adaptive"},
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
    )

    return boto3.client("bedrock-runtime", config=boto_config)


class BedrockClient:
    """
    Generic AWS Bedrock LLM client for making requests with images.
    
    This client supports:
    - Single and batch image processing
    - Parallel LLM invocations using ThreadPoolExecutor
    - Configurable model parameters via config.json
    """

    def __init__(self):
        """Initialize the Bedrock client with configuration from config.json."""
        self.llm_config = get_llm_config()
        self.client = get_bedrock_client()
        self.model_id = self.llm_config.get(
            "model_id", "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        self.max_tokens = self.llm_config.get("max_tokens", 65536)
        self.temperature = self.llm_config.get("temperature", 0)
        self.anthropic_version = self.llm_config.get(
            "anthropic_version", "bedrock-2023-05-31"
        )
        self.max_workers = self.llm_config.get("max_workers", 5)

    def prepare_model_request(
        self,
        user_prompt: str,
        images: Optional[list[tuple[bytes, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Prepare the request body for Bedrock model invocation.
        
        Args:
            user_prompt: The user prompt text
            images: Optional list of (image_bytes, media_type) tuples.
                    media_type should be like 'image/jpeg' or 'image/png'
            system_prompt: Optional system prompt for the model
            
        Returns:
            Request body dictionary ready for model invocation
        """
        # Build the content array for the user message
        content = []

        # Add images first (if any)
        if images:
            for image_bytes, media_type in images:
                # Encode image to base64
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                })

        # Add the text prompt
        content.append({"type": "text", "text": user_prompt})

        # Build the request body
        request_body = {
            "anthropic_version": self.anthropic_version,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }

        # Add system prompt if provided
        if system_prompt:
            request_body["system"] = system_prompt

        return request_body

    def invoke(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """
        Invoke the Bedrock model with the prepared request body.
        
        Args:
            request_body: The prepared request body from prepare_model_request
            
        Returns:
            Parsed response from the model
            
        Raises:
            Exception: If the model invocation fails
        """
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            return response_body

        except Exception as e:
            logger.error(f"Bedrock invocation failed: {str(e)}")
            raise

    def invoke_with_prompt(
        self,
        user_prompt: str,
        images: Optional[list[tuple[bytes, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Convenience method to prepare and invoke in one call.
        
        Args:
            user_prompt: The user prompt text
            images: Optional list of (image_bytes, media_type) tuples
            system_prompt: Optional system prompt for the model
            
        Returns:
            Parsed response from the model
        """
        request_body = self.prepare_model_request(
            user_prompt=user_prompt,
            images=images,
            system_prompt=system_prompt,
        )
        return self.invoke(request_body)

    def invoke_parallel(
        self,
        requests: list[dict[str, Any]],
        max_workers: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Invoke multiple requests in parallel using ThreadPoolExecutor.
        
        Args:
            requests: List of request dictionaries, each containing:
                - 'user_prompt': str
                - 'images': Optional[list[tuple[bytes, str]]]
                - 'system_prompt': Optional[str]
                - 'metadata': Optional[dict] - will be included in response
            max_workers: Number of parallel workers (defaults to config value)
            
        Returns:
            List of response dictionaries with results and metadata
        """
        workers = max_workers or self.max_workers
        results = []

        def process_single_request(request_data: dict, index: int) -> dict:
            """Process a single request and return result with metadata."""
            try:
                user_prompt = request_data.get("user_prompt", "")
                images = request_data.get("images")
                system_prompt = request_data.get("system_prompt")
                metadata = request_data.get("metadata", {})

                response = self.invoke_with_prompt(
                    user_prompt=user_prompt,
                    images=images,
                    system_prompt=system_prompt,
                )

                return {
                    "index": index,
                    "success": True,
                    "response": response,
                    "metadata": metadata,
                    "error": None,
                }
            except Exception as e:
                logger.error(f"Parallel request {index} failed: {str(e)}")
                return {
                    "index": index,
                    "success": False,
                    "response": None,
                    "metadata": request_data.get("metadata", {}),
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_request, req, idx): idx
                for idx, req in enumerate(requests)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                result = future.result()
                results.append(result)

        # Sort by original index to maintain order
        results.sort(key=lambda x: x["index"])

        return results

    def extract_text_response(self, response: dict[str, Any]) -> str:
        """
        Extract the text content from a Bedrock response.
        
        Args:
            response: The parsed response from invoke()
            
        Returns:
            The text content from the response
        """
        try:
            content = response.get("content", [])
            for item in content:
                if item.get("type") == "text":
                    return item.get("text", "")
            return ""
        except Exception as e:
            logger.error(f"Failed to extract text from response: {str(e)}")
            return ""

