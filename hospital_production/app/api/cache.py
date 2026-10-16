import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

class InferenceCache:
    """Thread-safe LRU + TTL cache for prediction results"""

    def __init__(self, max_size: int = 10_000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def make_key(payload: dict) -> str:
        """Generate a hash key based on the input payload"""
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache if it exists"""
        with self._lock:
            item = self._store.get(key)

            if not item:
                return None

            ts, value = item
            if time.time() - ts > self.ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)  # Mark as recently used
            return value

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache"""
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)  # Mark as recently used
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)  # Remove least recently used item

    def size(self) -> int:
        return len(self._store)

# Define _cache as a module-level variable
_cache: InferenceCache | None = None

def get_cache() -> InferenceCache:
    global _cache
    if _cache is None:
        from app.api.config import settings
        _cache = InferenceCache(settings.CACHE_MAX_SIZE, settings.CACHE_TTL_SECONDS)
    return _cache