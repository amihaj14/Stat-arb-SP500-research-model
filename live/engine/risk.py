# simple risk checks, limits


MAX_NOTIONAL_PER_PAIR = 10000
MAX_OPEN_PAIRS = 5
MAX_PCT_CAPITAL_PER_PAIR = 0.2
STOP_LOSS_MULTIPLIER = 1.5
MAX_DRAWDOWN_PER_PAIR = 0.325

def new_position(positions_dict, capital, size_notional):
    if len(positions_dict) >= MAX_OPEN_PAIRS:
        return False, "Too many open pairs"

    if size_notional > MAX_NOTIONAL_PER_PAIR:
        return False, "Requested size exceeds per-pair notional limit"

    if size_notional > MAX_PCT_CAPITAL_PER_PAIR * capital:
        return False, "Requested size exceeds per-pair capital fraction"

    return True, ""

def stop_loss(position, latest_pnl, current_z, entry_z):
    if latest_pnl['drawdown'] <= -MAX_DRAWDOWN_PER_PAIR:
        return True, "Pair drawdown beyond limit"
    elif abs(current_z) >= STOP_LOSS_MULTIPLIER * entry_z:
        return True, f"Stop loss hit at {position}"
    return False, ""


def map_zscore_to_side(z_score, entry_z, exit_z, current_side):
    if current_side is None:
        if z_score >= entry_z:
            return "short_spread"
        elif z_score <= -entry_z:
            return "long_spread"
        else:
            return None
    else:
        if -exit_z < z_score < exit_z:
            return None
        else:
            return current_side 