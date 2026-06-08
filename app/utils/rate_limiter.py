import time
import logging
from collections import defaultdict
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    def __init__(self):
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def check(self, key: str, max_requests: int, window_seconds: int = 60) -> None:
        now = time.time()
        window_start = now - window_seconds
        buckets = self._buckets[key]
        buckets[:] = [t for t in buckets if t > window_start]

        if len(buckets) >= max_requests:
            oldest = buckets[0] if buckets else now
            retry_after = int(window_seconds - (now - oldest))
            logger.warning(f"Rate limit excedido para {key}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Muitas requisições. Aguarde antes de tentar novamente.",
                    "retry_after_seconds": max(retry_after, 1),
                },
            )

        buckets.append(now)

    async def get_remaining(self, key: str, max_requests: int, window_seconds: int = 60) -> int:
        now = time.time()
        window_start = now - window_seconds
        buckets = self._buckets.get(key, [])
        active = [t for t in buckets if t > window_start]
        return max(0, max_requests - len(active))


rate_limiter = InMemoryRateLimiter()

AUDIO_RATE_LIMIT = 10
AUDIO_WINDOW = 60
CHAT_VOICE_RATE_LIMIT = 10
CHAT_VOICE_WINDOW = 60
