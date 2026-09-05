import csv
import itertools
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fixtures import DatasetFixture
from fm_adaptation import neurite_agreement as agreement
from fm_adaptation.neurite_annotations import identity, mapping, canonical_annotator
from fm_adaptation.report import (_interactive_neurite_table, _render_neurite_agreement,
                                  _render_table, _pool_folds, _mean_sd, _median_iqr)


YVONNE = "Dataset300_neurite_yvonne_smi"
B2 = "Dataset301_neurite_yvonne_b2_smi"
PAIR = "Yvonne__Cond-Lesion-GelMa-rods-Rat-13-slide1-section-6"
JS = Path(__file__).resolve().parents[1] / "src/fm_adaptation/assets/neurite_report.js"


class AnnotationTests(unittest.TestCase):
    def test_workbook_counts_raters_and_scale_variants(self):
        self.assertEqual(Counter(mapping()["Yvonne"]["cases"].values()), {"Coco": 5, "Yvonne": 5, "Tanya": 3})
        self.assertEqual(Counter(mapping()["Yvonne_b2"]["cases"].values()), {"Queena": 11, "Sarah": 2})
        self.assertEqual(identity(PAIR + "_rater1")[1], "Coco")
        self.assertEqual(identity(PAIR + "_rater2_scale150")[1], "Tanya")
        self.assertEqual(canonical_annotator("Cocco"), "Coco")

    def test_legacy_dataset203_is_second_batch(self):
        old = "Cond Lesion GelMa rods-Rat 3 slide1 section 3-GFAP SERT SMI PDGFRa_scale125"
        self.assertEqual(identity(old)[:2], ("Yvonne_b2", "Queena"))
        self.assertIsNone(identity("Yvonne_in_vitro__another-image"))
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            identity("Yvonne__missing")


class HumanAgreementTests(unittest.TestCase):
    def test_pair_across_splits_cache_tolerances_and_fixed_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = DatasetFixture(root, YVONNE, {"0": "SMI"})
            a = np.zeros((20, 20), dtype=np.uint8)
            b = a.copy()
            a[5:15, 5], b[5:15, 7] = 1, 1
            data.add(PAIR + "_rater1", split="Tr", label=a)
            data.add(PAIR + "_rater2", split="Ts", label=b)
            data.add(PAIR + "_rater1", split="Ts_interrater", label=a)
            data.add(PAIR + "_rater1_scale125", split="Tr", label=a)
            rows = agreement.measure(data.path)
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["annotator_a"], rows[0]["annotator_b"]), ("Coco", "Tanya"))
            self.assertEqual(rows[0]["dice"], 0)
            self.assertEqual(rows[0]["cldice"], 0)
            self.assertEqual(rows[0]["cldice_2px"], 1)
            self.assertIn("measured 1", agreement.cache(data.path, root))
            stored = agreement.read(agreement.path_for(root, YVONNE))
            self.assertEqual(stored, rows)
            with patch.object(agreement, "measure", side_effect=AssertionError("cache must be reused")):
                self.assertIn("up to date", agreement.cache(data.path, root))
            page = _render_neurite_agreement([YVONNE], root, "Mean ± SD", 2)
            self.assertIn("clDice@2px", page)
            self.assertIn("Coco | Tanya", page)
            self.assertIn("100.00", page)
            self.assertIn("independent of the annotator checkboxes", page)
            # Removing a tolerance invalidates the cache even if the label signature is unchanged.
            path = agreement.path_for(root, YVONNE)
            path.write_text("annotator_a,annotator_b,image,dice\nCoco,Tanya,image,0\n")
            page = _render_neurite_agreement([YVONNE], root, "Median (Q1–Q3)", 4)
            self.assertIn("Selected clDice tolerance not cached", page)
            self.assertIn("measured 1", agreement.cache(data.path, root))

    def test_unpaired_empty_and_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = DatasetFixture(root, YVONNE, {"0": "SMI"})
            data.add(PAIR + "_rater1")
            self.assertEqual(agreement.measure(data.path), [])
            agreement.cache(data.path, root)
            self.assertIn("No paired annotations available", _render_neurite_agreement([YVONNE], root, "Mean ± SD", 0))
            data.add(PAIR + "_rater2", split="Ts", label=np.zeros((3, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "matching native dimensions"):
                agreement.measure(data.path)


@unittest.skipUnless(shutil.which("node"), "Node is required to verify offline report calculations")
class InteractiveTests(unittest.TestCase):
    def node(self, program, payload):
        result = subprocess.run(["node", "-e", "const api=require(process.argv[1]);"
                                 "let input=JSON.parse(require('fs').readFileSync(0,'utf8'));" + program,
                                 str(JS)], input=json.dumps(payload), text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def records(self):
        def values(cases, scores):
            return {"cases": np.array(cases), "dice": np.array(scores), "cldice_2px": np.array(scores) * 0.9}
        cases = [PAIR + "_rater1", PAIR + "_rater2", "Yvonne__Cond-Lesion-GelMa-rods-Rat-1-slide11-section-1"]
        b2 = ["Yvonne_b2__Cond-Lesion-GelMa-rods-Rat-3-slide1-section-3",
              "Yvonne_b2__Cond-Lesion-GelMa-rods-Rat-6-slide11-section-1"]
        records = {}
        for model, scores in (("A", [0.2, 0.8, 0.6]), ("B", [0.9, 0.3, np.nan])):
            for fold in ("0", "1"):
                records[(model, "", YVONNE, fold)] = {
                    YVONNE: values(cases, scores), B2: values(b2, [0.4, 0.7]),
                    "Dataset999_neurite_other": values(["Other__1"], [0.5]),
                }
        # Missing metric columns are padded on pooling, as with older model caches.
        del records[("B", "", YVONNE, "1")][B2]["cldice_2px"]
        return _pool_folds(records, ["0", "1"])

    def test_js_matches_python_tables_for_independent_selections_and_rankings(self):
        records = self.records()
        datasets = [YVONNE, B2, "Dataset999_neurite_other"]
        metric_names = ("dice", "cldice_2px")
        # An in-domain column stays outside both the ranking and cross-dataset average.
        reference = {(YVONNE, YVONNE)}
        jobs, expected = [], []
        selections = [dict(Yvonne=y, Yvonne_b2=b) for y, b in itertools.product(
            [[], ["Coco"], ["Yvonne", "Tanya"], ["Coco", "Yvonne", "Tanya"]],
            [[], ["Queena"], ["Sarah"], ["Queena", "Sarah"]])]
        for statistic, grouped in itertools.product(["Mean ± SD", "Median (Q1–Q3)"], [False, True]):
            page = _interactive_neurite_table(records, datasets, statistic, lambda key: key,
                                              reference, metric_names, grouped)
            payload = json.loads(re.search(r'<script type="application/json" class="neurite-data">(.*?)</script>', page).group(1))
            for selected in selections:
                filtered = {}
                for key, results in records.items():
                    filtered[key] = {}
                    for dataset, values in results.items():
                        keep = []
                        for case in values["cases"]:
                            who = identity(case)
                            keep.append(who is None or who[1] in selected[who[0]])
                        filtered[key][dataset] = {metric: array[keep] for metric, array in values.items()}
                table = _render_table(filtered, datasets, statistic, lambda key: key,
                                      reference, metric_names, grouped, show_counts=True)
                cells = [re.findall(r'<td[^>]*>(.*?)</td>', row)[5:]
                         for row in re.findall(r'<tr[^>]*>(.*?)</tr>', table.split('<tbody>')[1])]
                expected.append([[cell.replace("'", '"') for cell in row] for row in cells])
                jobs.append({"data": payload, "selected": selected})
        actual = self.node("process.stdout.write(JSON.stringify(input.map(job=>api.calculate(job.data,job.selected,true)"
                           ".map(row=>row.map(cell=>api.cellHTML(cell,job.data.statistic))))));", jobs)
        self.assertEqual(actual, expected)

    def test_python_number_formatting_undefined_and_singletons(self):
        values = [[], [None], [0.00125], [0.00625], [0.2, None, 0.8], [0.312456, 0.72, 0.894567, 0.112345]]
        for statistic, fmt in [("Mean ± SD", _mean_sd), ("Median (Q1–Q3)", _median_iqr)]:
            actual = self.node("process.stdout.write(JSON.stringify(input.values.map(v=>api.format(v,100,input.statistic))));",
                               {"values": values, "statistic": statistic})
            self.assertEqual(actual, [fmt([np.nan if x is None else x for x in row], 100).replace("'", '"') for row in values])

    def test_payload_escapes_script_end_and_rejects_unknown_cases(self):
        records = self.records()
        key = next(iter(records))
        page = _interactive_neurite_table({("</script>", *key[1:]): records[key]}, [YVONNE],
                                         "Mean ± SD", lambda key: key, (), ("dice",), False)
        self.assertIn("&lt;/script&gt;", page)
        records[key][YVONNE]["cases"][0] = "Yvonne__unknown"
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            _interactive_neurite_table(records, [YVONNE], "Mean ± SD", lambda key: key, (), ("dice",), False)
