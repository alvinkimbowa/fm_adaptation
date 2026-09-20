import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fm_adaptation.config import ExperimentConfig, PromptConfig, PromptField
from fm_adaptation.data import NnUNet2DDataset, collate_cases, prompt_batch, prompt_classes
from fm_adaptation.prompting import CategoricalPromptEncoder
from fixtures import DatasetFixture, preprocess


LOCATION = PromptField(key="location", vocabulary=("wrist", "elbow"), dropout_p=0.5)
ANATOMY = PromptField(
    key="anatomy", vocabulary=("nerve", "artery", "bone"), multi=True, eval_values=("nerve",)
)


class VocabularyTests(unittest.TestCase):
    """How an answer becomes a row of embedding indices."""

    def test_each_field_opens_its_own_block_with_its_own_null(self):
        prompt = PromptConfig(fields=(LOCATION, ANATOMY))
        self.assertEqual(prompt.offsets, (0, 3))
        self.assertEqual(prompt.rows, 3 + 4)
        self.assertEqual(prompt.slots, 1 + 3)
        self.assertEqual(prompt.row({"location": ("elbow",), "anatomy": ("bone", "nerve")}),
                         (2, 6, 4, 3))
        self.assertEqual(prompt.row({}), (0, 3, 3, 3))

    def test_a_value_outside_the_vocabulary_is_the_same_as_none(self):
        prompt = PromptConfig(fields=(LOCATION,))
        self.assertEqual(prompt.row({"location": ("unspecified",)}), prompt.row({}))

    def test_one_field_is_the_table_a_run_without_fields_had(self):
        prompt = PromptConfig(fields=(PromptField(key="location", vocabulary=("a", "b", "c")),))
        encoder = CategoricalPromptEncoder(prompt, 8)
        self.assertEqual(list(encoder.state_dict()), ["embedding.weight"])
        self.assertEqual(tuple(encoder.embedding.weight.shape), (4, 8))
        rows = torch.tensor([[0], [2]])
        self.assertTrue(torch.equal(encoder(rows), encoder(rows.squeeze(1))))


class EncoderTests(unittest.TestCase):
    """What the lookup table does with a row of indices."""

    def setUp(self):
        torch.manual_seed(0)
        self.prompt = PromptConfig(fields=(LOCATION, ANATOMY))
        self.encoder = CategoricalPromptEncoder(self.prompt, 8)

    def encode(self, **values):
        return self.encoder(torch.tensor([self.prompt.row(values)]))

    def test_a_set_is_the_mean_of_its_entries(self):
        table = self.encoder.embedding.weight
        both = self.encode(anatomy=("nerve", "artery"))
        expected = (table[4] + table[5]) / 2 + table[0]
        self.assertTrue(torch.allclose(both[0], expected, atol=1e-6))

    def test_padding_does_not_dilute_the_answer(self):
        one = self.encode(anatomy=("nerve",))
        table = self.encoder.embedding.weight
        self.assertTrue(torch.allclose(one[0], table[4] + table[0], atol=1e-6))

    def test_dropping_one_field_leaves_the_other_untouched(self):
        with_location = self.encode(location=("wrist",), anatomy=("nerve",))
        without = self.encode(anatomy=("nerve",))
        self.assertTrue(torch.allclose(
            with_location - without, self.encode(location=("wrist",)) - self.encode(), atol=1e-6
        ))


class SamplingTests(unittest.TestCase):
    """What the dataset asks for, and what it therefore expects back."""

    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = DatasetFixture(
            self.root, "Dataset900_prompt", {"0": "RGB"},
            labels={"background": 0, "nerve": 1, "artery": 2, "bone": 3},
        )
        # One frame with three structures traced, one with a single structure.
        label = np.zeros((16, 12), dtype=np.uint8)
        label[0:4], label[4:8], label[8:12] = 1, 2, 3
        self.data.add("all", color=np.zeros((16, 12, 3), np.uint8), label=label)
        self.data.add("one", color=np.zeros((16, 12, 3), np.uint8), label=(label == 1).astype(np.uint8))
        self.data.split(["all"], ["one"])
        self.data.metadata({
            "all": {"location": "elbow", "anatomy": ["nerve", "artery", "bone"]},
            "one": {"location": "wrist", "anatomy": ["nerve"]},
        })
        self.prompt = PromptConfig(fields=(LOCATION, ANATOMY))

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, subset):
        return NnUNet2DDataset(
            self.root, self.data.name, "Tr", "0", subset, preprocess, prompt=self.prompt,
        )

    def test_training_only_ever_asks_for_structures_the_frame_has(self):
        train = self.dataset("train")
        seen = set()
        for _ in range(200):
            asked = train._ask("all")["anatomy"]
            self.assertTrue(set(asked) <= {"nerve", "artery", "bone"})
            self.assertTrue(asked)
            seen.add(tuple(sorted(asked)))
        self.assertEqual(len(seen), 7)

    def test_the_target_holds_what_was_asked_for_and_nothing_else(self):
        train = self.dataset("train")
        train.ids = ["all"] * 60
        for index in range(60):
            _, mask, metadata = train[index]
            present = set(torch.unique(mask).tolist()) - {0}
            self.assertTrue(present <= set(metadata["classes"]))
            self.assertEqual(set(metadata["classes"]) - {0}, present)

    def test_validation_asks_the_fixed_question_and_ignores_dropout(self):
        val = self.dataset("val")
        for _ in range(20):
            self.assertEqual(val._ask("one"), {"location": ("wrist",), "anatomy": ("nerve",)})
        _, mask, metadata = val[0]
        self.assertEqual(metadata["classes"], (1,))
        self.assertEqual(metadata["prompt"], self.prompt.row(
            {"location": ("wrist",), "anatomy": ("nerve",)}
        ))

    def test_a_batch_carries_the_union_of_what_its_cases_asked_for(self):
        train = self.dataset("train")
        train.ids = ["all", "one"]
        _, _, metadata = collate_cases([train[0], train[1]])
        self.assertEqual(tuple(prompt_batch(metadata).shape), (2, self.prompt.slots))
        union = set(metadata[0]["classes"]) | set(metadata[1]["classes"])
        self.assertEqual(prompt_classes(metadata), sorted(union))

    def test_labels_the_dataset_does_not_declare_leave_the_target_alone(self):
        # Predicting a 702-trained model on 701 reads a dataset whose labels are named otherwise;
        # there is nothing to restrict against and nothing that needs restricting.
        other = DatasetFixture(self.root, "Dataset901_binary", {"0": "RGB"})
        label = np.ones((16, 12), dtype=np.uint8)
        other.add("case", color=np.zeros((16, 12, 3), np.uint8), label=label)
        other.split(["case"])
        other.metadata({"case": {"location": "wrist", "anatomy": "nerve"}}, nested=False)
        dataset = NnUNet2DDataset(
            self.root, other.name, "Tr", "0", "train", preprocess, prompt=self.prompt,
        )
        self.assertIsNone(dataset.prompt_labels)
        _, mask, metadata = dataset[0]
        self.assertNotIn("classes", metadata)
        self.assertTrue(bool((mask == 1).all()))


class ConfigTests(unittest.TestCase):
    """What a config may say about a prompt."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        DatasetFixture(self.root, "Dataset900_prompt", {"0": "RGB"})

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, prompt):
        body = {
            "data": {"raw_data_dir": str(self.root), "train_dataset": "Dataset900_prompt",
                     "fold": 0, "prompt": prompt},
            "model": {"name": "dinov3", "probe": "upernet"},
            "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001},
        }
        path = self.root / "config.yaml"
        path.write_text(json.dumps(body))
        return path

    def test_a_single_key_and_vocabulary_becomes_one_field(self):
        cfg = ExperimentConfig.from_yaml(self.write({"key": "location", "vocabulary": ["a", "b"]}))
        self.assertEqual(cfg.prompt.fields,
                         (PromptField(key="location", vocabulary=("a", "b"), dropout_p=0.2),))

    def test_fields_and_the_single_form_cannot_both_be_given(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_yaml(self.write(
                {"vocabulary": ["a"], "fields": [{"key": "anatomy", "vocabulary": ["b"]}]}
            ))

    def test_eval_must_name_entries_of_its_own_field(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_yaml(self.write(
                {"fields": [{"key": "anatomy", "vocabulary": ["a"], "eval": ["b"]}]}
            ))

    def test_the_same_key_cannot_be_asked_twice(self):
        with self.assertRaises(ValueError):
            ExperimentConfig.from_yaml(self.write({"fields": [
                {"key": "anatomy", "vocabulary": ["a"]},
                {"key": "anatomy", "vocabulary": ["b"]},
            ]}))


if __name__ == "__main__":
    unittest.main()
