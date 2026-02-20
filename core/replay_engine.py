from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReplayTick:
    timestamp: float
    symbol: str
    price: float


class ReplayEngine:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def load_ticks(self) -> list[ReplayTick]:
        ticks: list[ReplayTick] = []
        with self.csv_path.open('r', encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                ticks.append(ReplayTick(float(row['timestamp']), str(row['symbol']), float(row['price'])))
        return ticks
