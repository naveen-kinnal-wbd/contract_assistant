"""
Generic AWS Bedrock LLM client for making requests with images.

This module provides a reusable client for interacting with AWS Bedrock,
supporting image-based prompts with document metadata.
"""

import base64
import json
import logging
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

    This client supports building requests with multiple images,
    each annotated with document filename and page number.
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

    def prepare_request(
        self,
        prompt: str,
        images: list[dict[str, Any]],
        schema: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Prepare the request body for Bedrock model invocation.

        Args:
            prompt: The user prompt text
            images: List of image dicts with keys:
                - bytes: The raw image bytes
                - media_type: e.g., "image/jpeg"
                - filename: Document filename
                - page: Page number
            schema: Optional schema dict to embed in the prompt as JSON

        Returns:
            Request body dictionary ready for model invocation
        """
        # Embed schema into prompt if provided
        if schema is not None:
            prompt = prompt.replace("{INPUT_SCHEMA}", json.dumps(schema, indent=4))

        # Build content array starting with the text prompt
        content = [{"type": "text", "text": prompt}]

        # Add images with captions
        for img in images:
            image_b64 = base64.b64encode(img["bytes"]).decode("utf-8")

            # Add image
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": image_b64,
                    },
                }
            )

            # Add image caption
            content.append(
                {
                    "type": "text",
                    "text": f"Document: {img['identifier_name']} - Page {img['page_number']}",
                }
            )

        # Build the request body
        request_body = {
            "anthropic_version": self.anthropic_version,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }

        return request_body

    def invoke(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """
        Invoke the Bedrock model with the prepared request body.

        Args:
            request_body: The prepared request body from prepare_request

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

    def extract_text_response(self, response: dict[str, Any]) -> str:
        """
        Extract the text content from a Bedrock response.

        Args:
            response: The parsed response from invoke()

        Returns:
            The text content from the response
        """
        content = response.get("content", [])
        for item in content:
            if item.get("type") == "text":
                return item.get("text", "")
        return ""

    def extract_json_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Extract and parse JSON content from a Bedrock response.

        Handles markdown code block formatting (```json ... ```) that LLMs
        commonly include in their responses.

        Args:
            response: The parsed response from invoke()

        Returns:
            Parsed JSON as a dictionary, or empty dict if parsing fails
        """
        # Extract text content from response
        result = self.extract_text_response(response)
        if not result:
            logger.warning("No text content found in Bedrock response")
            return {}

        # Strip whitespace
        result = result.strip()

        # Remove markdown code block formatting if present
        if result.startswith("```json"):
            result = result.replace("```json", "", 1)
        elif result.startswith("```"):
            result = result.replace("```", "", 1)

        if result.endswith("```"):
            result = result[:-3]

        # Parse the JSON
        try:
            return json.loads(result.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {result[:500]}...")
            return {}
