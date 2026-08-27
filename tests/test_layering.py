"""Predicting, tabulating and drawing are three jobs, and each has to stand on its own.

They are free to share a utility -- `naming` and `selection` exist for exactly that -- but neither
the code nor the comments of one may be written in terms of another's output. A comment explaining a
figure by what a table does with it goes stale the moment the table changes, and nothing fails.
"""

import re
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "src" / "fm_adaptation"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# Modules that produce predictions or figures. None of them decides how a table is laid out.
INDEPENDENT = (
    "predict.py",
    "robustness.py",
    "compare_qualitative.py",
    "plot_qualitative.py",
    "qualitative.py",
)


class LayeringTests(unittest.TestCase):
    def test_nothing_but_the_table_imports_the_table(self):
        """`count_params.py` is exempt: it exists to fill the table's parameter column, so it is
        part of that concern rather than another one reaching across."""
        offenders = [
            path.name
            for path in SOURCE.glob("*.py")
            if path.name not in {"report.py", "count_params.py"}
            and re.search(r"from \.report import", path.read_text())
        ]
        self.assertEqual(offenders, [], "shared naming belongs in naming.py, not report.py")

    def test_prediction_and_figure_logic_is_not_explained_by_the_table(self):
        pattern = re.compile(r"results table|main table|report\.py|the report\b", re.I)
        offenders = []
        for path in [SOURCE / name for name in INDEPENDENT] + sorted(SCRIPTS.glob("*qualitative.sh")):
            offenders += [
                f"{path.name}:{number}"
                for number, line in enumerate(path.read_text().splitlines(), 1)
                if pattern.search(line)
            ]
        self.assertEqual(offenders, [])

    def test_the_two_figure_scripts_do_not_import_each_other(self):
        for name in ("compare_qualitative.py", "plot_qualitative.py"):
            other = "plot_qualitative" if name.startswith("compare") else "compare_qualitative"
            self.assertNotIn(f"from .{other} import", (SOURCE / name).read_text())


if __name__ == "__main__":
    unittest.main()
