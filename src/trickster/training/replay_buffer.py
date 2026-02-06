"""Simple circular replay buffer for experience replay training."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

Experience = Tuple[Dict[str, float], float]  # (features, target_y)


class ReplayBuffer:
    """Fixed-capacity circular buffer of (features, y) pairs.

    Oldest entries are overwritten once capacity is reached.
    """

    __slots__ = ("_buf", "_capacity", "_pos", "_size")

    def __init__(self, capacity: int = 50_000) -> None:
        self._buf: list[Experience] = [({}, 0.0)] * capacity
        self._capacity = capacity
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, features: Dict[str, float], y: float) -> None:
        self._buf[self._pos] = (features, y)
        self._pos = (self._pos + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def sample(self, n: int, rng: random.Random) -> List[Experience]:
        """Return *n* uniformly-sampled experiences (with replacement if n > size)."""
        if self._size == 0:
            return []
        indices = [rng.randrange(self._size) for _ in range(n)]
        return [self._buf[i] for i in indices]
