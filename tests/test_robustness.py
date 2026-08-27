import tempfile
import unittest
from pathlib import Path

import torch

from fm_adaptation.data import STAIN_PLANES, NnUNet2DDataset
from fixtures import DatasetFixture, preprocess
try:
    from fm_adaptation.robustness import TRANSFORMS, Transform, apply, invert, smi_cases, table
except ImportError as error:  # pragma: no cover - environment, not behaviour
    # Scoring needs monai, which only .venv-mm carries; the figure tests need scikit-image, which
    # only the SAM3 environment has. Skipping keeps one interpreter from failing to collect what the
    # other checks.
    raise unittest.SkipTest(f"fm_adaptation.robustness unavailable here: {error}") from error


FILL = [-2.0, -2.0, -2.0]


def _block(top=8, left=40, size=6, canvas=64):
    """One image with a bright block off-centre, and the label map that marks it."""
    image = torch.full((1, 3, canvas, canvas), -2.0)
    image[:, :, top : top + size, left : left + size] = 1.0
    prediction = torch.zeros((1, canvas, canvas), dtype=torch.long)
    prediction[:, top : top + size, left : left + size] = 1
    return image, prediction


class TransformTests(unittest.TestCase):
        """What the input transform does, and that the prediction comes back where it started."""

        def test_a_flip_is_its_own_inverse(self):
            _, prediction = _block()
            for transform in (Transform("h", hflip=True), Transform("v", vflip=True)):
                turned = invert(
                    torch.flip(prediction, [-1] if transform.hflip else [-2]), transform,
                )
                self.assertTrue(torch.equal(turned, prediction), transform.name)

        def test_a_rotation_and_a_zoom_return_the_block_to_where_it_started(self):
            """Not pixel-exact -- two nearest-neighbour resamples move a boundary -- but in place."""
            _, prediction = _block()
            before = prediction[0].nonzero().float().mean(0)
            for transform in (
                Transform("r", degrees=10.0),
                Transform("s", scale=1.25),
                Transform("rs", degrees=-10.0, scale=0.75),
            ):
                turned = TF_forward(prediction, transform)
                self.assertFalse(torch.equal(turned, prediction), transform.name)
                back = invert(turned, transform)
                after = back[0].nonzero().float().mean(0)
                self.assertLess(float((after - before).abs().max()), 2.0, transform.name)

        def test_dropping_smi_clears_that_plane_and_leaves_the_other_alone(self):
            image, _ = _block()
            image[:, STAIN_PLANES["GFAP"]] = 0.5
            dropped = apply(image, Transform("drop", drop_smi=True), FILL)
            self.assertTrue(torch.all(dropped[:, STAIN_PLANES["SMI"]] == FILL[0]))
            self.assertTrue(torch.all(dropped[:, STAIN_PLANES["GFAP"]] == 0.5))

        def test_a_case_whose_smi_is_already_blank_is_not_counted_as_carrying_it(self):
            """Eric's sections ship an empty SMI file, so a column of them cannot move."""
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = DatasetFixture(root, "Dataset998_smi", {"0": "SMI", "1": "GFAP"})
                data.add("Mohammad__both", planes={0: 31, 1: 47})
                data.add("Eric__gfap_only", planes={0: 0, 1: 47})
                data.split(["Mohammad__both", "Eric__gfap_only"])
                dataset = NnUNet2DDataset(root, data.name, "Tr", "0", "eval", preprocess)
                self.assertEqual(smi_cases(dataset), 1)

        def test_a_model_never_given_smi_has_none_to_drop(self):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = DatasetFixture(root, "Dataset997_gfap", {"0": "SMI", "1": "GFAP"})
                data.add("Mohammad__both", planes={0: 31, 1: 47})
                data.split(["Mohammad__both"])
                dataset = NnUNet2DDataset(
                    root, data.name, "Tr", "0", "eval", preprocess, keep_planes=frozenset({2}),
                )
                self.assertEqual(smi_cases(dataset), 0)

        def test_the_identity_transform_leaves_the_image_untouched(self):
            """The `none` row has to reproduce the main table, so it must not resample anything."""
            image, _ = _block()
            self.assertTrue(torch.equal(apply(image, TRANSFORMS[0], FILL), image))

        def test_every_transform_has_a_distinct_usable_directory_name(self):
            names = [t.directory for t in TRANSFORMS]
            self.assertEqual(len(set(names)), len(names))
            for name in names:
                self.assertNotIn(" ", name)
                self.assertNotIn("/", name)


class TableTests(unittest.TestCase):
        def test_the_change_block_is_measured_against_none(self):
            rows = [
                ("none", {"Dataset214_lesion_mohammad_smi_gfap": 0.800}),
                ("h-flip", {"Dataset214_lesion_mohammad_smi_gfap": 0.500}),
            ]
            rendered = table(rows, ["Dataset214_lesion_mohammad_smi_gfap"], {"Dataset214_lesion_mohammad_smi_gfap": (3, 4)}, "a run")
            self.assertIn("| 214 mohammad |", rendered)
            self.assertIn("-0.300", rendered)
            self.assertIn("3/4", rendered)

        def test_a_column_with_no_score_reads_as_missing_rather_than_as_zero(self):
            rows = [("none", {"Dataset212_lesion_katie_dorsal_column_smi_gfap": None})]
            rendered = table(
                rows, ["Dataset212_lesion_katie_dorsal_column_smi_gfap"],
                {"Dataset212_lesion_katie_dorsal_column_smi_gfap": (0, 5)}, "a run",
            )
            self.assertIn("| -- |", rendered)


def TF_forward(prediction, transform):
    """The geometry `apply` performs, on a label map, so the test can invert it."""
    from torchvision.transforms import functional as TF

    return TF.affine(
        prediction.unsqueeze(1).float(), angle=transform.degrees, translate=[0, 0],
        scale=transform.scale, shear=[0.0, 0.0],
        interpolation=TF.InterpolationMode.NEAREST, fill=[0.0],
    ).squeeze(1).long()


if __name__ == "__main__":
    unittest.main()
