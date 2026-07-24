import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from research.Strategy import lin_reg, z_score, generate_signals

WINDOW_LENGTH = 26          # ~1 trading day of 15-min bars
ENTRY_THRESHOLD = 2.5       # tighter than daily; tune via grid search below
EXIT_THRESHOLD = 0.5