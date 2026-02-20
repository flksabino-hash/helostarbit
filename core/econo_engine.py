from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict

import numpy as np

from utils import hurst_rs, shannon_entropy_returns, lyapunov_rosenstein


class EconoEngine:
    def __init__(self, maxlen: int = 2048) -> None:
        self.history: DefaultDict[str, deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, symbol: str, price: float) -> None:
        self.history[symbol].append(float(price))

    def _arr(self, symbol: str) -> np.ndarray:
        return np.asarray(self.history[symbol], dtype=float)

    def hurst(self, symbol: str) -> float:
        return float(hurst_rs(self._arr(symbol)))

    def entropy(self, symbol: str) -> float:
        return float(shannon_entropy_returns(self._arr(symbol)))

    def lle(self, symbol: str) -> float:
        return float(lyapunov_rosenstein(self._arr(symbol)))

    def amh_regime(self, symbol: str) -> str:
        h = self.hurst(symbol)
        e = self.entropy(symbol)
        l = self.lle(symbol)
        if e > 3.2 or h < 0.45 or l >= 0:
            return "contagion"
        if h > 0.55 and e < 2.8 and l < 0:
            return "trending"
        return "neutral"
