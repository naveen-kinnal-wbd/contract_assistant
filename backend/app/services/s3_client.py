"""
S3 client service for the contract assistant workflow.

This module provides a centralized S3 client with methods for:
- Fetching images from S3
- Uploading documents to S3
- Parsing S3 URIs
"""

from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from config import get_upload_config

# Boto3 config with connection pooling and reasonable timeouts
_boto_config = BotoConfig(
    connect_timeout=5,
    read_timeout=30,
    max_pool_connections=10,
    retries={"max_attempts": 3, "mode": "standard"},
)


@lru_cache(maxsize=1)
def get_s3_client():
    """Get cached S3 client with configured region and connection pooling."""
    upload_config = get_upload_config()
    region = upload_config.get("region", "us-east-1")
    return boto3.client("s3", region_name=region, config=_boto_config)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI into bucket and key.

    Args:
        s3_uri: S3 URI in format s3://bucket/key

    Returns:
        Tuple of (bucket, key)
    """
    path = s3_uri.replace("s3://", "")
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def build_upload_path(identifier_name: str, file_type: str) -> str:
    """
    Build the S3 upload path for a document.

    Args:
        identifier_name: Identifier name for the document
        file_type: Document file type classification

    Returns:
        S3 path in format: <path_prefix>/<identifier_name>/<file_type>/
    """
    upload_config = get_upload_config()
    path_prefix = upload_config.get("path_prefix", "uploaded")
    return f"{path_prefix}/{identifier_name}/{file_type}"


def fetch_image_from_s3(s3_uri: str) -> dict[str, Any]:
    """
    Fetch an image from S3 and return image data with metadata.

    The metadata (original_filename, page_number, media_type, file_type)
    is retrieved from the S3 object's user-defined metadata.

    Args:
        s3_uri: S3 URI of the image

    Returns:
        Dict with keys: bytes, media_type, identifier_name, page_number
    """
    s3_client = get_s3_client()
    bucket, key = parse_s3_uri(s3_uri)

    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()

    # Get user-defined metadata from S3 object
    s3_metadata = response.get("Metadata", {})

    # Use metadata from S3, with fallbacks based on file extension
    media_type = s3_metadata.get("media_type")
    if not media_type:
        if key.lower().endswith(".jpeg") or key.lower().endswith(".jpg"):
            media_type = "image/jpeg"
        elif key.lower().endswith(".png"):
            media_type = "image/png"
        else:
            media_type = "image/jpeg"

    identifier_name = s3_metadata.get("identifier_name", "")
    page_number = s3_metadata.get("page_number", "")

    return {
        "bytes": image_bytes,
        "media_type": media_type,
        "identifier_name": identifier_name,
        "page_number": page_number,
    }


def upload_to_s3(
    bucket: str,
    s3_key: str,
    content: bytes,
    content_type: str,
    metadata: dict[str, str],
) -> None:
    """
    Upload content to S3.

    Args:
        bucket: S3 bucket name
        s3_key: S3 object key
        content: Raw bytes to upload
        content_type: MIME type of the content
        metadata: User-defined metadata to attach to the object
    """
    s3_client = get_s3_client()
    s3_client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=content,
        ContentType=content_type,
        Metadata=metadata,
    )
