from __future__ import annotations

from pathlib import Path

from backtest import run_synthetic_backtest
from config import Settings, load_settings
from core.replay_engine import ReplayEngine
from core.runtime_engine import RuntimeEngine
from core.snapshot import save_snapshot_and_fills
from infra.structured_logging import configure_logging


def main() -> None:
    configure_logging('INFO')
    settings = load_settings(None)
    engine = RuntimeEngine(settings)

    # backtest/replay engine stays outside UI
    run_synthetic_backtest(settings, minutes=30)

    csv_path = Path(__file__).parent / 'sample_data' / 'replay_ticks.csv'
    replay = ReplayEngine(csv_path)
    for tick in replay.load_ticks():
        engine.bus.publish('tick', {'ts': tick.timestamp, 'sym': tick.symbol, 'price': tick.price, 'source': 'replay_csv'})

    snap_json, fills_csv = save_snapshot_and_fills(engine.wallet, engine.state.last_prices, Path(__file__).parent / 'artifacts' / 'snapshot_latest')
    print(f'Snapshot JSON: {snap_json}')
    print(f'Fills CSV: {fills_csv}')
    print(f'Final metrics keys: {list(engine.state.metrics.keys())}')


if __name__ == '__main__':
    main()
