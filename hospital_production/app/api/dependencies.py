import logging
from functools import lru_cache
from app.ml.registry import load_concept
from app.ml.multimodel import pick_best

logger = logging.getLogger(__name__)

@lru_cache(maxsize=8)
def get_model_bundle(concept: str):
    """Load a concept bundle from MinIO, cache in process"""
    bundle = load_concept(concept)
    best_name, best = pick_best(bundle["models"], bundle["task"])
    logger.info(f"✅ Loaded concept '{concept}' with best model '{best_name}'")

    return {
        "pipeline": best["pipeline"],
        "best_submodel": best_name,
        "features": bundle["features"],
        "task": bundle["task"]
    }

