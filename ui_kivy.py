"""
ui_kivy.py — optional read-only mobile UI (Kivy)

Shows:
- Equity / balance
- Open positions count
- Latest BTC/ETH/SOL price snapshot
Reads from: ./data/status.json written by main.py

Run:
  python ui_kivy.py
"""
from __future__ import annotations

import json
from pathlib import Path

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label


STATUS_PATH = Path("data/status.json")


class Root(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.lbl = Label(text="Waiting for status.json ...", halign="left", valign="top")
        self.lbl.bind(size=self._reflow)
        self.add_widget(self.lbl)
        Clock.schedule_interval(self.refresh, 1.0)

    def _reflow(self, *_):
        self.lbl.text_size = self.lbl.size

    def refresh(self, *_):
        if not STATUS_PATH.exists():
            return
        try:
            j = json.loads(STATUS_PATH.read_text())
            prices = j.get("prices", {})
            txt = (
                f"ts: {j.get('ts')}\n"
                f"equity: {j.get('equity'):.2f}\n"
                f"balance: {j.get('balance'):.2f}\n"
                f"open_positions: {j.get('open_positions')}\n\n"
                f"BTCUSDT: {prices.get('BTCUSDT')}\n"
                f"ETHUSDT: {prices.get('ETHUSDT')}\n"
                f"SOLUSDT: {prices.get('SOLUSDT')}\n"
            )
            self.lbl.text = txt
        except Exception:
            pass


class PolyScalperUI(App):
    def build(self):
        return Root()


if __name__ == "__main__":
    PolyScalperUI().run()
