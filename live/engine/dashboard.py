from flask import Flask, render_template_string, request
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

app = Flask(__name__)

STARTING_CAPITAL = {
    "daily": 100_000,
    "intraday": 100_000,
}

DATA_SOURCES = {
    "daily": {
        "label": "Daily",
        "pnl": Path("live/logs/daily_pnl.csv"),
        "trades": Path("live/logs/trades.csv"),
        "positions": Path("live/logs/positions.csv"),
    },
    "intraday": {
        "label": "Intraday (15m)",
        "pnl": Path("live/logs/daily_pnl_intraday.csv"),
        "trades": Path("live/logs/trades_intraday.csv"),
        "positions": Path("live/logs/positions_intraday.csv"),
    },
}

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Pairs Trader Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    background: #0f1115;
    color: #e6e6e6;
    margin: 0;
    padding: 20px;
  }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .timestamp { color: #888; font-size: 13px; margin-bottom: 20px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 24px; }
  .tab {
    padding: 8px 18px;
    border-radius: 8px;
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    color: #999;
    background: #1a1d24;
    border: 1px solid #2a2e38;
  }
  .tab.active { color: #fff; background: #2e63e8; border-color: #2e63e8; }
  .summary-row { display: flex; gap: 12px; margin-bottom: 28px; flex-wrap: wrap; }
  .card {
    background: #1a1d24;
    border: 1px solid #2a2e38;
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    min-width: 140px;
  }
  .card .label { font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 24px; font-weight: 600; margin-top: 4px; }
  .positive { color: #3ecf8e; }
  .negative { color: #ef5b5b; }
  .neutral { color: #e6e6e6; }
  section {
    background: #15171c;
    border: 1px solid #2a2e38;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 20px;
  }
  section h2 {
    font-size: 15px;
    margin: 0 0 12px 0;
    color: #ccc;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 11px;
    color: #999;
    margin-bottom: 14px;
  }
  .legend-gradient {
    width: 160px;
    height: 10px;
    border-radius: 5px;
    background: linear-gradient(to right, #2e63e8, #2a2e38, #d9364a);
  }
  .heatmap-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 10px;
  }
  .tile {
    border-radius: 8px;
    padding: 10px 12px;
    position: relative;
    border: 1px solid rgba(255,255,255,0.08);
    transition: transform 0.15s ease;
  }
  .tile:hover { transform: scale(1.03); }
  .tile .pair-name {
    font-size: 12px;
    font-weight: 600;
    color: #fff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
  }
  .tile .z-value {
    font-size: 20px;
    font-weight: 700;
    color: #fff;
    margin-top: 4px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
  }
  .tile .pnl-value {
    font-size: 11px;
    margin-top: 4px;
    color: rgba(255,255,255,0.85);
  }
  .tile .open-badge {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3ecf8e;
    box-shadow: 0 0 6px #3ecf8e;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid #23262e;
    white-space: nowrap;
  }
  th { color: #999; font-weight: 500; font-size: 11px; text-transform: uppercase; }
  tr:hover td { background: #1c1f26; }
  .empty-msg { color: #666; font-style: italic; padding: 8px 0; }
</style>
</head>
<body>
  <h1>Pairs Trader — Live Status</h1>
  <div class="timestamp">Last refreshed: {{ now }} (auto-refreshes every 60s)</div>

  <div class="tabs">
    <a class="tab {{ 'active' if view == 'daily' else '' }}" href="/?source=daily">Daily</a>
    <a class="tab {{ 'active' if view == 'intraday' else '' }}" href="/?source=intraday">Intraday (15m)</a>
  </div>

  <div class="summary-row">
    <div class="card">
      <div class="label">Current Capital</div>
      <div class="value {{ capital_class }}">${{ current_capital }}</div>
    </div>
    <div class="card">
      <div class="label">Open Positions</div>
      <div class="value neutral">{{ open_count }}</div>
    </div>
    <div class="card">
      <div class="label">Total Realized PnL</div>
      <div class="value {{ realized_class }}">${{ realized_pnl }}</div>
    </div>
    <div class="card">
      <div class="label">Total Unrealized PnL</div>
      <div class="value {{ unrealized_class }}">${{ unrealized_pnl }}</div>
    </div>
    <div class="card">
      <div class="label">Trades Today</div>
      <div class="value neutral">{{ trades_today }}</div>
    </div>
  </div>

  <section>
    <h2>Pair Z-Score Heatmap — {{ source_label }}</h2>
    <div class="legend">
      <span>-3.0 (long spread)</span>
      <div class="legend-gradient"></div>
      <span>+3.0 (short spread)</span>
      <span style="margin-left:14px;">🟢 dot = open position</span>
    </div>
    {{ heatmap|safe }}
  </section>

  <section>
    <h2>Open Positions</h2>
    {{ positions|safe }}
  </section>

  <section>
    <h2>Recent Trades</h2>
    {{ trades|safe }}
  </section>
</body>
</html>
"""

def safe_read_df(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except pd.errors.EmptyDataError:
        return None


def style_table(df: pd.DataFrame) -> str:
    if df is None:
        return '<div class="empty-msg">No data yet.</div>'
    return df.to_html(index=False, classes="styled", border=0, escape=False)


def zscore_to_color(z: float, cap: float = 3.0) -> str:
    z = max(-cap, min(cap, z))
    t = z / cap

    neutral = (42, 46, 56)
    blue = (46, 99, 232)
    red = (217, 54, 74)

    if t < 0:
        frac = -t
        r = neutral[0] + (blue[0] - neutral[0]) * frac
        g = neutral[1] + (blue[1] - neutral[1]) * frac
        b = neutral[2] + (blue[2] - neutral[2]) * frac
    else:
        frac = t
        r = neutral[0] + (red[0] - neutral[0]) * frac
        g = neutral[1] + (red[1] - neutral[1]) * frac
        b = neutral[2] + (red[2] - neutral[2]) * frac

    return f"rgb({int(r)}, {int(g)}, {int(b)})"


def build_heatmap(pnl_df: pd.DataFrame, pos_df: pd.DataFrame) -> str:
    if pnl_df is None or "pair_id" not in pnl_df.columns or "z_score" not in pnl_df.columns:
        return '<div class="empty-msg">No z-score data yet.</div>'

    latest = pnl_df.sort_values("timestamp").groupby("pair_id").tail(1)

    open_pairs = set()
    if pos_df is not None and "status" in pos_df.columns and "pair_id" in pos_df.columns:
        open_pairs = set(pos_df.loc[pos_df["status"] == "open", "pair_id"])

    tiles = []
    for _, row in latest.iterrows():
        pair_id = row["pair_id"]
        z = float(row["z_score"]) if pd.notna(row.get("z_score")) else 0.0
        unrealized = row.get("unrealized_pnl", 0.0)
        realized = row.get("realized_pnl", 0.0)
        pnl = unrealized if pd.notna(unrealized) and unrealized != 0 else realized
        pnl = pnl if pd.notna(pnl) else 0.0

        color = zscore_to_color(z)
        badge = '<div class="open-badge"></div>' if pair_id in open_pairs else ""

        tiles.append(f"""
        <div class="tile" style="background:{color};">
          {badge}
          <div class="pair-name">{pair_id}</div>
          <div class="z-value">{z:+.2f}</div>
          <div class="pnl-value">PnL: ${pnl:,.2f}</div>
        </div>
        """)

    return f'<div class="heatmap-grid">{"".join(tiles)}</div>'


@app.route("/")
def home():
    view = request.args.get("source", "daily")
    if view not in DATA_SOURCES:
        view = "daily"

    paths = DATA_SOURCES[view]
    pos_df = safe_read_df(paths["positions"])
    trades_df = safe_read_df(paths["trades"])
    pnl_df = safe_read_df(paths["pnl"])

    open_count = 0
    if pos_df is not None and "status" in pos_df.columns:
        open_count = (pos_df["status"] == "open").sum()

    realized_pnl = 0.0
    unrealized_pnl = 0.0
    if pnl_df is not None:
        if "realized_pnl" in pnl_df.columns:
            realized_pnl = pnl_df["realized_pnl"].sum()
        if "unrealized_pnl" in pnl_df.columns:
            latest_ts = pnl_df["timestamp"].max() if "timestamp" in pnl_df.columns else None
            if latest_ts is not None:
                unrealized_pnl = pnl_df.loc[pnl_df["timestamp"] == latest_ts, "unrealized_pnl"].sum()
            else:
                unrealized_pnl = pnl_df["unrealized_pnl"].sum()

    trades_today = 0
    if trades_df is not None and "timestamp" in trades_df.columns:
        today = datetime.now(timezone.utc).date()
        trades_df["_date"] = pd.to_datetime(trades_df["timestamp"]).dt.date
        trades_today = (trades_df["_date"] == today).sum()
        trades_df = trades_df.drop(columns=["_date"]).tail(20)

    current_capital = STARTING_CAPITAL[view] + realized_pnl + unrealized_pnl
    starting_capital_for_view = STARTING_CAPITAL[view]
    
    heatmap_html = build_heatmap(pnl_df, pos_df)

    return render_template_string(
        TEMPLATE,
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        view=view,
        source_label=DATA_SOURCES[view]["label"],
        heatmap=heatmap_html,
        positions=style_table(pos_df),
        trades=style_table(trades_df),
        open_count=open_count,
        current_capital=f"{current_capital:,.2f}",
        capital_class="positive" if current_capital > starting_capital_for_view else ("negative" if current_capital < starting_capital_for_view else "neutral"),
        realized_pnl=f"{realized_pnl:,.2f}",
        unrealized_pnl=f"{unrealized_pnl:,.2f}",
        realized_class="positive" if realized_pnl > 0 else ("negative" if realized_pnl < 0 else "neutral"),
        unrealized_class="positive" if unrealized_pnl > 0 else ("negative" if unrealized_pnl < 0 else "neutral"),
        trades_today=trades_today,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)