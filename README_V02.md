# Whale Hunter v0.2 (CRISPR)

## Added in this cut
- Feed layer pluggable (`feeds/`) with `ReplayCSVFeedPlugin` and `PolymarketGammaCLOBFeedPlugin`.
- `core/runtime_engine.py` decoupled from UI.
- `core/econo_engine.py` with Hurst / Entropy / LLE + AMH regime classifier.
- `core/venue_selector.py`, `core/event_bus.py`, `core/replay_engine.py`.
- JSON structlog configuration (`infra/structured_logging.py`).
- Snapshot JSON + fills CSV export helper.
- Real unittest coverage (`tests/test_v02_core.py`).

## Quick run
```bash
python run_v02_replay.py
python -m unittest tests/test_v02_core.py -v
```
