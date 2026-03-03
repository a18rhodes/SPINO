"""
Persistent cache for MOSFET physics parameters to avoid redundant SPICE extraction.

Implements multiprocessing-safe disk-based caching using SQLite to share
parameter tensors across DataLoader workers and training runs.
"""

import pickle
import sqlite3
from pathlib import Path

import torch

__all__ = ["PhysicsCache"]


class PhysicsCache:
    """
    Multiprocessing-safe persistent cache for device physics parameters.

    Uses SQLite for atomic transactions and file locking to safely share
    cached parameter tensors across multiple DataLoader worker processes.
    """

    def __init__(self, cache_dir: str | Path = "/tmp/spino_physics_cache"):
        """
        Initializes cache with specified storage directory.

        :param cache_dir: Directory path for SQLite database storage.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "params.db"
        self._init_database()

    def _init_database(self):
        """Creates cache table and enables WAL mode for concurrent reads."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS physics_cache (
                    cache_key TEXT PRIMARY KEY,
                    tensor_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON physics_cache(created_at)")
            conn.commit()

    def get(self, model_name: str, width: float, length: float) -> torch.Tensor | None:
        """
        Retrieves cached physics tensor for device geometry.

        :param model_name: SPICE model identifier.
        :param width: Transistor width in microns.
        :param length: Transistor length in microns.
        :return: Cached tensor or None if not found.
        """
        key = f"{model_name}_w{width}_l{length}"
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("SELECT tensor_data FROM physics_cache WHERE cache_key = ?", (key,))
                if row := cursor.fetchone():
                    return pickle.loads(row[0])
        except sqlite3.Error:
            return None
        return None

    def put(self, model_name: str, width: float, length: float, tensor: torch.Tensor):
        """
        Stores physics tensor in cache with atomic transaction.

        :param model_name: SPICE model identifier.
        :param width: Transistor width in microns.
        :param length: Transistor length in microns.
        :param tensor: Physics parameter tensor to cache.
        """
        key = f"{model_name}_w{width}_l{length}"
        tensor_blob = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO physics_cache (cache_key, tensor_data) VALUES (?, ?)",
                    (key, tensor_blob),
                )
                conn.commit()
        except sqlite3.Error:
            pass

    def size(self) -> int:
        """
        Returns number of cached entries.

        :return: Cache entry count.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM physics_cache")
                return cursor.fetchone()[0]
        except sqlite3.Error:
            return 0
