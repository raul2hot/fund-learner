"""
Rate limiter with exponential backoff for API requests.
"""

import time
import random
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')


class RateLimiter:
    """
    Rate limiter with exponential backoff.

    Features:
    - Respects rate limits per time window
    - Automatic exponential backoff on failures
    - Jitter to prevent thundering herd
    """

    def __init__(
        self,
        requests_per_minute: int = 1200,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Track request timing
        self._request_times: list = []
        self._min_interval = 60.0 / requests_per_minute

    def _clean_old_requests(self):
        """Remove requests older than 1 minute."""
        current_time = time.time()
        self._request_times = [
            t for t in self._request_times
            if current_time - t < 60.0
        ]

    def wait_if_needed(self):
        """Wait if we're approaching rate limit."""
        self._clean_old_requests()

        if len(self._request_times) >= self.requests_per_minute:
            oldest = self._request_times[0]
            wait_time = 60.0 - (time.time() - oldest) + 0.1
            if wait_time > 0:
                time.sleep(wait_time)
                self._clean_old_requests()

    def record_request(self):
        """Record that a request was made."""
        self._request_times.append(time.time())

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function with rate limiting and retry logic.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result of function call

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                self.wait_if_needed()
                result = func(*args, **kwargs)
                self.record_request()
                return result

            except Exception as e:
                last_exception = e

                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(
                        self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                        self.max_delay
                    )
                    print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)

        raise last_exception


def rate_limited(limiter: RateLimiter):
    """Decorator to apply rate limiting to a method."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return limiter.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator
