# position & PnL tracking
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from live.engine.notify import notify

POSITION_PATH = Path("live/logs/positions.csv")
TRADES_PATH = Path("live/logs/trades.csv")
PNL_PATH = Path("live/logs/daily_pnl.csv")


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_positions():
    if not POSITION_PATH.exists():
        return {}

    try:
        df = pd.read_csv(POSITION_PATH)
    except pd.errors.EmptyDataError:
        return {}

    if df.empty or "status" not in df.columns:
        return {}

    pos = {}
    for _, row in df[df["status"] == "open"].iterrows():
        pos[row["pair_id"]] = row.to_dict()
    return pos


def save_positions(positions_dict):
    POSITION_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not positions_dict:
        # write an empty file with just the header so future reads don't choke,
        # or simply skip writing anything if the file doesn't exist yet
        if POSITION_PATH.exists():
            return
        pd.DataFrame(columns=[
            "pair_id", "s1", "s2", "beta", "side", "size_notional",
            "qty_s1", "qty_s2", "entry_price_s1", "entry_price_s2",
            "entry_time", "status", "realized_pnl", "exit_time",
        ]).to_csv(POSITION_PATH, index=False)
        return
    
    df = pd.DataFrame(positions_dict.values())
    df.to_csv(POSITION_PATH, index=False)

def log_trade(trade_row: dict):
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TRADES_PATH.exists():
        try:
            df = pd.read_csv(TRADES_PATH)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([trade_row])], ignore_index=True)
    else:
        df = pd.DataFrame([trade_row])
    df.to_csv(TRADES_PATH, index=False)

    notify(
        f"{trade_row['pair_id']}: {trade_row['action'].upper()} "
        f"{trade_row.get('side', '')} — {trade_row.get('reason', '')}",
        title=f"Trade {trade_row['action']}",
        priority="high" if trade_row["action"] == "close" else "default",
    )


def log_pnl(pnl_row: dict):
    PNL_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = PNL_PATH.exists() and PNL_PATH.stat().st_size > 0
    pd.DataFrame([pnl_row]).to_csv(PNL_PATH, mode="a", header=not file_exists, index=False)


def open_spread(pair_cfg, latest_prices, side: str, size_notional: float):
    s1, s2 = pair_cfg["s1"], pair_cfg["s2"]
    p1, p2 = latest_prices[s1], latest_prices[s2]
    qty_s1 = size_notional / p1
    qty_s2 = size_notional / p2

    if side == "long_spread":  # long s2, short s1
        qty_s1 = -qty_s1
    elif side == "short_spread":  # short s2, long s1
        qty_s2 = -qty_s2

    pos = {
        "pair_id": pair_cfg["pair_id"],
        "s1": s1,
        "s2": s2,
        "beta": pair_cfg["beta"],
        "side": side,
        "size_notional": size_notional,
        "qty_s1": qty_s1,
        "qty_s2": qty_s2,
        "entry_price_s1": p1,
        "entry_price_s2": p2,
        "entry_time": _utc_now_iso(),
        "status": "open",
        "realized_pnl": 0.0,
        "exit_time": None,
    }
    return pos


def close_spread(position, latest_prices):
    s1, s2 = position["s1"], position["s2"]
    p1, p2 = latest_prices[s1], latest_prices[s2]
    qty_s1, qty_s2 = position["qty_s1"], position["qty_s2"]

    pnl = qty_s1 * (p1 - position["entry_price_s1"]) + qty_s2 * (p2 - position["entry_price_s2"])

    position["status"] = "closed"
    position["exit_time"] = _utc_now_iso()
    position["realized_pnl"] = pnl
    return position, pnl
