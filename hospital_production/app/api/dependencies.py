import logging
from functools import lru_cache
from app.ml.registry import load_concept
from app.ml.multimodel import pick_best
from app.api.metrics import MODEL_LOADED

logger = logging.getLogger(__name__)

@lru_cache(maxsize=8)
def get_model_bundle(concept: str):
    """Load a concept bundle from MinIO, cache in process"""
    try:
        bundle = load_concept(concept)        
        MODEL_LOADED.labels(model_name=concept).set(1)
        return bundle
    
    except Exception as exc:                    

        logging.warning(
            f"⚠️ Model bundle '{concept}' not available in MinIO – "
            f"service will continue without it. Reason: {exc}"
        )
        
        MODEL_LOADED.labels(model_name=concept).set(0)
        return None