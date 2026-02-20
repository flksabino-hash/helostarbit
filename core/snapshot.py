from __future__ import annotations

import csv
import json
from pathlib import Path

from risk_manager import PaperWallet


def save_snapshot_and_fills(wallet: PaperWallet, mark_prices: dict[str, float], base_path: str | Path) -> tuple[Path, Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix('.json')
    csv_path = base.with_suffix('.csv')

    snapshot = {
        'balance_usdc': wallet.balance_usdc,
        'equity': wallet.equity(mark_prices),
        'drawdown_pct': None,
        'positions': {k: {'shares': v} for k, v in wallet.inv_shares.items()},
        'fills_count': len(wallet.fills),
    }
    json_path.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ts', 'order_id', 'token_id', 'side', 'size_shares', 'price', 'fee_usdc'])
        for fill in wallet.fills:
            writer.writerow([fill.ts, fill.order_id, fill.token_id, fill.side, fill.size_shares, fill.price, fill.fee_usdc])

    return json_path, csv_path
