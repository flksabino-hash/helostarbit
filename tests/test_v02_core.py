from __future__ import annotations

import unittest
from pathlib import Path

from core.econo_engine import EconoEngine
from core.event_bus import EventBus
from core.replay_engine import ReplayEngine
from core.runtime_engine import RuntimeEngine


class TestV02Core(unittest.TestCase):
    def test_event_bus(self) -> None:
        bus = EventBus()
        got: list[int] = []
        bus.subscribe('x', lambda p: got.append(int(p['v'])))
        bus.publish('x', {'v': 42})
        self.assertEqual(got, [42])

    def test_econo_metrics(self) -> None:
        e = EconoEngine()
        px = 0.45
        for i in range(250):
            px += 0.002 if i % 9 < 6 else -0.001
            e.update('MKT', px)
        self.assertIsInstance(e.hurst('MKT'), float)
        self.assertIsInstance(e.entropy('MKT'), float)
        self.assertIsInstance(e.lle('MKT'), float)
        self.assertIn(e.amh_regime('MKT'), {'contagion', 'neutral', 'trending'})

    def test_replay_runtime_smoke(self) -> None:
        engine = RuntimeEngine()
        csv_path = Path(__file__).resolve().parents[1] / 'sample_data' / 'replay_ticks.csv'
        replay = ReplayEngine(csv_path)
        for t in replay.load_ticks()[:140]:
            engine.bus.publish('tick', {'ts': t.timestamp, 'sym': t.symbol, 'price': t.price, 'source': 'replay_csv'})
        self.assertIn('BTC_100K_MAR', engine.state.last_prices)
        self.assertIn('BTC_100K_MAR', engine.state.metrics)


if __name__ == '__main__':
    unittest.main(verbosity=2)
