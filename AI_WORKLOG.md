# AI Worklog

## 2026-02-20T00:00:00Z - v0.2 CRISPR skeleton
- Preservado o cérebro legado (config, exchange, strategy, risk_manager, backtest, etc.) na raiz do projeto.
- Adicionada arquitetura modular v0.2 com `core/`, `feeds/`, `infra/`, `tests/` e `sample_data/`.
- Implementado feed layer pluggável com Replay CSV e Polymarket Gamma/CLOB (plugin) publicando no EventBus.
- Implementados `EconoEngine` (Hurst/Entropy/LLE), `AMH regime classifier`, `VenueSelector` e `RuntimeEngine` desacoplado da UI.
- Adicionado snapshot JSON + CSV de fills e testes reais (unittest) para feed/runtime.
- Corrigido bug em `utils.py` (faltava `import re`).
