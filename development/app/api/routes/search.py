"""
Search Routes - FastAPI endpoints for computer vision image search
"""

import logging
import os
import tempfile
from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from api.services.cv_search import CVSearchService, get_cv_search

# Configure logging & router
logger = logging.getLogger(__name__)
router = APIRouter()

# ==========================================
# Schemas
# ==========================================


class SearchResult(BaseModel):
    brand: str
    path: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query_image: str


class BrandsResponse(BaseModel):
    brands: List[str]
    total: int


class StatsResponse(BaseModel):
    total_images: int
    feature_dimension: int
    brands: int


# ==========================================
# Endpoints
# ==========================================
# Configure allowed extensions for image uploads
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}


@router.post(
    "/similar",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search similar cars by image",
)
async def search_similar(
    file: UploadFile = File(...),
    k: int = 5,
    cv_service: CVSearchService = Depends(get_cv_search),
):
    """Search for similar cars using an uploaded image file."""
    # Extract file extension and validate
    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"

    # Validate againts either MIME header or file extension
    is_valid_mime = file.content_type and file.content_type.startswith("image/")
    is_valid_ext = suffix in ALLOWED_EXTENSIONS

    if not (is_valid_mime or is_valid_ext):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="❌ Invalid file format. Uploaded file must be an image.",
        )

    tmp_path = None  # Initialize tmp_path for cleanup in finally block
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        results = cv_service.search_by_image(tmp_path, k)

        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"❌ No similar cars found matching the query image ({file.filename}).",
            )

        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            total=len(results),
            query_image=file.filename or "uploaded_image.jpg",
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Visual search failed unexpectedly.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="❌ An error occurred while processing the image search.",
        ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get(
    "/brands",
    response_model=BrandsResponse,
    summary="List available car brands",
)
async def list_brands(
    cv_service: CVSearchService = Depends(get_cv_search),
):
    """List all available car brands indexed in the system."""
    try:
        brands = cv_service.get_available_brands()
        return BrandsResponse(brands=brands, total=len(brands))
    except Exception as exc:
        logger.exception("Failed to retrieve brands.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="❌ Failed to retrieve brand list.",
        ) from exc


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get search service statistics",
)
async def search_stats(
    cv_service: CVSearchService = Depends(get_cv_search),
):
    """Get indexing and feature statistics for the search service."""
    try:
        stats = cv_service.get_stats()
        return StatsResponse(**stats)
    except Exception as exc:
        logger.exception("Failed to fetch search stats.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="❌ Failed to retrieve search statistics.",
        ) from exc
