from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VenueScore:
    name: str
    score: float


class VenueSelector:
    def choose(self, *, funding: float, oi_delta: float, volatility: float, margin_health: float, liquidity: float) -> VenueScore:
        score = (
            0.30 * float(liquidity)
            + 0.20 * float(oi_delta)
            + 0.20 * float(margin_health)
            - 0.15 * abs(float(funding))
            - 0.15 * float(volatility)
        )
        return VenueScore(name="paper" if score < 0.5 else "live", score=score)
