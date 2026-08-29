"""
Search Routes - FastAPI endpoints for computer vision image search
Supports both local and production CV search services
"""
import asyncio
import logging
import os
import tempfile
from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================
# Environment Configuration
# ============================================
USE_PRODUCTION = os.getenv("USE_PRODUCTION_SERVICES", "false").lower() == "true"

# ============================================
# Schemas
# ============================================
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

class DamageDetectionResponse(BaseModel):
    status: str
    damage_info: dict
    query_image: str

# ============================================
# CV Search Factory
# ============================================
def get_cv_search():
    """Get the appropriate CV search service based on environment"""
    if USE_PRODUCTION:
        try:
            from api.services.cv_search_production import get_cv_search as get_prod_cv_search
            cv_search = get_prod_cv_search()
            logger.info("✅ Using Production CV Search (MinIO)")
            return cv_search
        
        except Exception as e:
            logger.warning(f"⚠️ Production CV search failed: {e}. Falling back to local.")

    from api.services.cv_search import get_cv_search as get_local_cv_search
    logger.info("💻 Using Local CV Search")
    return get_local_cv_search()

# ============================================
# Allowed Extensions
# ============================================
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

# ============================================
# Endpoints
# ============================================
@router.post(
    "/similar",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search similar cars by image",
)
async def search_similar(
    file: UploadFile = File(...),
    k: int = 5,
):
    """Search for similar cars using an uploaded image file."""
    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"

    is_valid_mime = file.content_type and file.content_type.startswith("image/")
    is_valid_ext = suffix in ALLOWED_EXTENSIONS

    if not (is_valid_mime or is_valid_ext):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="❌ Invalid file format. Uploaded file must be an image.",
        )

    tmp_path = None
    cv_service = get_cv_search()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Use the service's search method
        if hasattr(cv_service, 'search_by_image'):
            results = cv_service.search_by_image(tmp_path, k)
        elif hasattr(cv_service, 'search'):
            results = cv_service.search(tmp_path, k)
        else:
            results = cv_service.search_similar(tmp_path, k) if hasattr(cv_service, 'search_similar') else []

        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"❌ No similar cars found matching the query image ({file.filename})."
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
async def list_brands():
    """List all available car brands indexed in the system."""
    try:
        cv_service = get_cv_search()
        
        if hasattr(cv_service, 'get_available_brands'):
            brands = cv_service.get_available_brands()
        elif hasattr(cv_service, 'feature_data'):
            brands = list(set([d['brand'] for d in cv_service.feature_data]))
        else:
            brands = []
        
        return BrandsResponse(brands=sorted(brands), total=len(brands))
    
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
async def search_stats():
    """Get indexing and feature statistics for the search service."""
    try:
        cv_service = get_cv_search()
        
        if hasattr(cv_service, 'get_stats'):
            stats = cv_service.get_stats()
        elif hasattr(cv_service, 'index') and hasattr(cv_service, 'metadata'):
            stats = {
                'total_images': cv_service.index.ntotal,
                'feature_dimension': cv_service.metadata.get('feature_dimension', 0),
                'brands': len(set([d['brand'] for d in cv_service.feature_data])) if hasattr(cv_service, 'feature_data') else 0
            }
        else:
            stats = {
                'total_images': 0,
                'feature_dimension': 0,
                'brands': 0
            }
        
        return StatsResponse(**stats)
    
    except Exception as exc:
        logger.exception("Failed to fetch search stats.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="❌ Failed to retrieve search statistics.",
        ) from exc

@router.get("/status")
async def get_search_status():
    """Get search service status"""
    try:
        cv_service = get_cv_search()
        source = "Production (MinIO)" if USE_PRODUCTION else "Local"
        return {
            "status": "healthy",
            "source": source,
            "environment": "production" if USE_PRODUCTION else "development",
            "stats": await search_stats() if hasattr(cv_service, 'get_stats') else {}
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/damage_detect", 
             response_model=DamageDetectionResponse, 
             summary="Detect damage in car image")
async def detect_damage(file: UploadFile = File(...)):
    """Detect damage in the uploaded car image."""
    suffix = Path(file.filename or ".jpg").suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="❌ Invalid file format. Uploaded file must be an image.",
        )

    tmp_path: str | None = None

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="❌ Uploaded file is empty.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        cv_service = get_cv_search()

        if not hasattr(cv_service, 'detect_damage'):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="❌ Damage detection is not implemented in the current CV search service.",
            )

        # Prevent blocking FastAPI's event loop
        damage_info = await asyncio.to_thread(cv_service.detect_damage, tmp_path)
        damage_info = damage_info or {}

        return DamageDetectionResponse(
            status="damage detected" if damage_info else "no damage detected",
            damage_info=damage_info,
            query_image=file.filename or "uploaded_image.jpg",
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Damage detection failed unexpectedly.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="❌ An error occurred while processing the damage detection.",
        ) from exc

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")