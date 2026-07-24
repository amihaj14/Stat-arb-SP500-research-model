# intraday paper-trading loop
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
import os
import sys

from live.engine.auth import get_session
from live.engine.live_data import get_latest_quotes, get_history
from live.engine.positions_intraday import (
    load_positions, save_positions, log_pnl, log_trade, open_spread, close_spread,
)
from live.engine.strategy_live import get_live_strategy_signal
from live.engine.risk import new_position, stop_loss, map_zscore_to_side, MAX_NOTIONAL_PER_PAIR
from live.engine.notify import notify

PNL_PATH = Path('live/logs/daily_pnl_intraday.csv')
PAIRS_CONFIG_PATH = Path('live/config/pairs_config_intraday.csv')
STARTING_CAPITAL = 100_000
REQUIRED_COLUMNS = [
    'Pair',
    'Stock1',
    'Stock2',
    'Alpha',
    'Beta',
    'Entry Threshold',
    'Exit Threshold',
    'Window Length',
]

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
ET = ZoneInfo("America/New_York")


def load_pairs_config():
    df = pd.read_csv(PAIRS_CONFIG_PATH)
    return validate_pairs_config(df)

def _get_current_capital(pnl_path, starting_capital=STARTING_CAPITAL):
    if not Path(pnl_path).exists():
        return starting_capital
    try:
        df = pd.read_csv(pnl_path)
    except pd.errors.EmptyDataError:
        return starting_capital
    if df.empty or "realized_pnl" not in df.columns:
        return starting_capital
    total_realized = df["realized_pnl"].fillna(0).sum()
    return starting_capital + total_realized

def validate_pairs_config(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError('Pairs config is empty')

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    normalized = df.copy()
    normalized = normalized.loc[:, REQUIRED_COLUMNS]

    for col in ['Alpha', 'Beta', 'Entry Threshold', 'Exit Threshold', 'Window Length']:
        normalized[col] = pd.to_numeric(normalized[col], errors='coerce')

    invalid_rows = normalized[
        normalized[['Pair', 'Stock1', 'Stock2']].isna().any(axis=1)
        | normalized[['Alpha', 'Beta', 'Entry Threshold', 'Exit Threshold', 'Window Length']].isna().any(axis=1)
        | (normalized['Window Length'] <= 0)
    ]

    if not invalid_rows.empty:
        invalid_indices = invalid_rows.index.tolist()
        raise ValueError(f'Invalid pairs config rows: {invalid_indices}')

    return normalized


def _is_market_hours(as_of_ts: datetime) -> bool:
    et_ts = as_of_ts.astimezone(ET)
    if et_ts.weekday() >= 5:
        return False
    t = et_ts.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE

def _lookback_days_for_window(window_length: int, bars_per_day: int = 26, buffer_days: int = 3) -> int:
    trading_days_needed = -(-window_length // bars_per_day)  # ceil division
    calendar_buffer = int(trading_days_needed * 1.5) + buffer_days  # 1.5x covers weekends
    return max(calendar_buffer, 5)


def _compute_unrealized_pnl_and_drawdown(position, latest_prices):
    s1, s2 = position["s1"], position["s2"]
    p1, p2 = latest_prices[s1], latest_prices[s2]
    qty_s1, qty_s2 = position["qty_s1"], position["qty_s2"]

    entry_p1 = position["entry_price_s1"]
    entry_p2 = position["entry_price_s2"]

    unrealized = qty_s1 * (p1 - entry_p1) + qty_s2 * (p2 - entry_p2)

    basis = abs(qty_s1 * entry_p1) + abs(qty_s2 * entry_p2)
    drawdown = unrealized / basis if basis > 0 else 0.0

    return {
        "unrealized_pnl": unrealized,
        "drawdown": drawdown,
    }


def paper_trade_step_intraday(as_of_ts: datetime):
    if not _is_market_hours(as_of_ts):
        print(f"{as_of_ts.isoformat()}: outside market hours, skipping intraday run.")
        return

    api_server, token = get_session()

    pairs_cfg = load_pairs_config()
    positions = load_positions()

    tickers = sorted(set(pairs_cfg["Stock1"]).union(set(pairs_cfg["Stock2"])))
    quotes_df = get_latest_quotes(tickers, api_server, token)
    latest_prices = quotes_df["lastTradePriceTrHrs"]

    max_window = int(pairs_cfg["Window Length"].max())
    # 15-min bars: pull enough lookback in *days* to cover max_window bars,
    # padded generously since only ~26 bars occur per trading day.
    lookback_days = _lookback_days_for_window(max_window)
    history = get_history(
        tickers,
        api_server,
        token,
        interval="FifteenMinutes",
        lookback_days=lookback_days,
    )

    for _, row in pairs_cfg.iterrows():
        pair_id = row["Pair"]
        s1 = row["Stock1"]
        s2 = row["Stock2"]
        beta = float(row["Beta"])
        entry_z = float(row["Entry Threshold"])
        exit_z = float(row["Exit Threshold"])
        window_length = int(row["Window Length"])

        if s1 not in history.columns or s2 not in history.columns:
            continue

        pair_prices = history[[s1, s2]].dropna()
        if len(pair_prices) < window_length:
            print(f"{pair_id}: skipping, only {len(pair_prices)} bars, need {window_length}")
            continue

        strat_config = {
            "Entry Threshold": entry_z,
            "Exit Threshold": exit_z,
            "Window Length": window_length,
            "Beta": beta,
        }

        signal_info = get_live_strategy_signal(
            prices=pair_prices,
            pair=pair_id,
            stock1=s1,
            stock2=s2,
            config=strat_config,
            beta=beta,
        )
        latest_z = signal_info["latest_z_score"]

        current_pos = positions.get(pair_id)
        current_side = current_pos["side"] if current_pos else None

        desired_side = map_zscore_to_side(latest_z, entry_z, exit_z, current_side)

        if current_pos is not None:
            latest_pnl = _compute_unrealized_pnl_and_drawdown(current_pos, latest_prices)
            should_stop, stop_reason = stop_loss(current_pos, latest_pnl, latest_z, entry_z)

            if should_stop or desired_side is None:
                closed_pos, realized = close_spread(current_pos, latest_prices)
                positions[pair_id] = closed_pos

                log_trade({
                    "timestamp": as_of_ts.isoformat(),
                    "pair_id": pair_id,
                    "action": "close",
                    "reason": stop_reason or "exit_band",
                    "side": current_side,
                    "price_s1": latest_prices[s1],
                    "price_s2": latest_prices[s2],
                    "realized_pnl": realized,
                })

                log_pnl({
                    "timestamp": as_of_ts.isoformat(),
                    "pair_id": pair_id,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": realized,
                    "drawdown": latest_pnl["drawdown"],
                    "z_score": latest_z,
                })

            else:
                log_pnl({
                    "timestamp": as_of_ts.isoformat(),
                    "pair_id": pair_id,
                    "unrealized_pnl": latest_pnl["unrealized_pnl"],
                    "realized_pnl": current_pos.get("realized_pnl", 0.0),
                    "drawdown": latest_pnl["drawdown"],
                    "z_score": latest_z,
                })

        else:
            if desired_side is None:
                log_pnl({
                    "timestamp": as_of_ts.isoformat(),
                    "pair_id": pair_id,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "drawdown": 0.0,
                    "z_score": latest_z,
                })
                continue

            size_notional = MAX_NOTIONAL_PER_PAIR
            ok, reason = new_position(positions, _get_current_capital(PNL_PATH), size_notional)
            if not ok:
                log_pnl({
                    "timestamp": as_of_ts.isoformat(),
                    "pair_id": pair_id,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "drawdown": 0.0,
                    "z_score": latest_z,
                })
                continue

            pair_cfg = {
                "pair_id": pair_id,
                "s1": s1,
                "s2": s2,
                "beta": beta,
            }
            pos = open_spread(pair_cfg, latest_prices, desired_side, size_notional)
            positions[pair_id] = pos

            log_trade({
                "timestamp": as_of_ts.isoformat(),
                "pair_id": pair_id,
                "action": "open",
                "reason": reason or "signal_entry",
                "side": desired_side,
                "price_s1": latest_prices[s1],
                "price_s2": latest_prices[s2],
                "notional": size_notional,
            })

            log_pnl({
                "timestamp": as_of_ts.isoformat(),
                "pair_id": pair_id,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "drawdown": 0.0,
                "z_score": latest_z,
            })

    save_positions(positions)

    et_ts = as_of_ts.astimezone(ET)
    if et_ts.hour == 9 and et_ts.minute <= 35:
        notify("Intraday trader ran successfully this morning.", title="Heartbeat (Intraday)")


LOCK_FILE = os.path.expanduser("~/.qt_trader_intraday.lock")


if os.path.exists(LOCK_FILE):
    print("Another intraday instance appears to be running. Exiting.")
    sys.exit(1)

try:
    open(LOCK_FILE, "w").close()
    if __name__ == "__main__":
        ts = datetime.now(timezone.utc)
        try:
            paper_trade_step_intraday(ts)
        except Exception as e:
            notify(f"Intraday trader crashed: {e}", title="ERROR (Intraday)", priority="urgent")
            raise
finally:
    os.remove(LOCK_FILE)