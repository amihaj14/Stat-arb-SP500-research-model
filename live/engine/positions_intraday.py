import live.engine.positions as _base
from pathlib import Path

_base.POSITION_PATH = Path("live/logs/positions_intraday.csv")
_base.TRADES_PATH = Path("live/logs/trades_intraday.csv")
_base.PNL_PATH = Path("live/logs/daily_pnl_intraday.csv")

load_positions = _base.load_positions
save_positions = _base.save_positions
log_pnl = _base.log_pnl
log_trade = _base.log_trade
open_spread = _base.open_spread
close_spread = _base.close_spread