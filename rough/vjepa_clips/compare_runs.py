"""Training curves of several runs on one pair of axes, drawn the way each run's own history.png is.

    PYTHONPATH=src python rough/vjepa_clips/compare_runs.py OUT.png TITLE RUN_DIR [RUN_DIR ...]

Each RUN_DIR is a fold directory holding a history.csv.
"""

import sys
from pathlib import Path

from fm_adaptation.plot_history import _plot, _read_history


def main():
    out_path, title, *run_dirs = sys.argv[1:]
    curves = []
    for run_dir in map(Path, run_dirs):
        history = _read_history(run_dir / "history.csv")
        if history.get("epoch"):
            curves.append((run_dir.parent.name, history))
    _plot(curves, title, Path(out_path))
    print(f"wrote {out_path} ({len(curves)} runs)")


if __name__ == "__main__":
    main()
