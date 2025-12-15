"""
Data caching utilities.

Stores API responses as Parquet files to avoid re-fetching during development.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
from typing import Optional, Dict, Any


class DataCache:
    """
    Cache for API responses and intermediate data.

    Features:
    - Stores data as Parquet files for efficient storage
    - Supports TTL (time-to-live) for cache invalidation
    - Generates unique cache keys based on request parameters
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        default_ttl_hours: int = 24
    ):
        """
        Initialize DataCache.

        Args:
            cache_dir: Directory for cache files
            default_ttl_hours: Default TTL in hours (default: 24)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)

        # Metadata file
        self.metadata_path = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _generate_key(self, **kwargs) -> str:
        """Generate unique cache key from parameters."""
        key_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        source: str,
        symbol: str = None,
        start: str = None,
        end: str = None,
        **extra_params
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and not expired.

        Args:
            source: Data source name
            symbol: Trading symbol
            start: Start date
            end: End date
            **extra_params: Additional parameters for cache key

        Returns:
            DataFrame if cache hit, None if miss or expired
        """
        params = {
            'source': source,
            'symbol': symbol,
            'start': start,
            'end': end,
            **extra_params
        }
        cache_key = self._generate_key(**params)

        # Check metadata
        if cache_key not in self.metadata:
            return None

        entry = self.metadata[cache_key]

        # Check TTL
        cached_at = datetime.fromisoformat(entry['cached_at'])
        if datetime.utcnow() - cached_at > self.default_ttl:
            print(f"Cache expired for {source}")
            return None

        # Load from file
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if not cache_path.exists():
            return None

        print(f"Cache hit for {source}")
        return pd.read_parquet(cache_path)

    def set(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str = None,
        start: str = None,
        end: str = None,
        **extra_params
    ):
        """
        Cache data.

        Args:
            df: DataFrame to cache
            source: Data source name
            symbol: Trading symbol
            start: Start date
            end: End date
            **extra_params: Additional parameters for cache key
        """
        params = {
            'source': source,
            'symbol': symbol,
            'start': start,
            'end': end,
            **extra_params
        }
        cache_key = self._generate_key(**params)

        # Save to file
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        df.to_parquet(cache_path, index=False)

        # Update metadata
        self.metadata[cache_key] = {
            'source': source,
            'symbol': symbol,
            'start': start,
            'end': end,
            'cached_at': datetime.utcnow().isoformat(),
            'rows': len(df),
            'file': str(cache_path),
        }
        self._save_metadata()

        print(f"Cached {source}: {len(df)} rows")

    def invalidate(self, source: str = None):
        """
        Invalidate cache entries.

        Args:
            source: If provided, only invalidate this source
        """
        keys_to_remove = []

        for key, entry in self.metadata.items():
            if source is None or entry.get('source') == source:
                keys_to_remove.append(key)
                cache_path = Path(entry['file'])
                if cache_path.exists():
                    cache_path.unlink()

        for key in keys_to_remove:
            del self.metadata[key]

        self._save_metadata()
        print(f"Invalidated {len(keys_to_remove)} cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            Path(entry['file']).stat().st_size
            for entry in self.metadata.values()
            if Path(entry['file']).exists()
        )

        return {
            'entries': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'sources': list(set(e['source'] for e in self.metadata.values())),
        }
