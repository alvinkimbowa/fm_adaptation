import tempfile
import unittest
from pathlib import Path

from fm_adaptation.config import ExperimentConfig


class ConfigTests(unittest.TestCase):
        """The real experiment configs, which the runs are launched from."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def test_real_configs_parse_and_defaults(self):
            repo = Path(__file__).resolve().parents[1]
            paths = sorted((repo / "configs").glob("dinov3_upernet_inj_ft*sci208.yaml"))
            configs = {ExperimentConfig.from_yaml(path).run_name: ExperimentConfig.from_yaml(path)
                       for path in paths}
            self.assertTrue(all(cfg.fold == "0" for cfg in configs.values()))
            self.assertTrue(all(cfg.channel_dropout_p == 0.5 for cfg in configs.values()))
            # One run per intervention, so a difference between two of them has a single cause.
            expected = {
                "upernet_inj_ft_ours": ((), False),
                "upernet_inj_ft_dropsmi_ours": (("SMI",), False),
                "upernet_inj_ft_dropany_ours": (("SMI", "GFAP"), False),
                "upernet_inj_ft_balanced_ours": ((), True),
                "upernet_inj_ft_balanced_dropany_ours": (("SMI", "GFAP"), True),
                "upernet_inj_ft_balanced_aug_ours": ((), True),
                "upernet_inj_ft_balanced_dropsmi_aug_ours": (("SMI",), True),
            }
            self.assertEqual(set(configs), set(expected))
            for name, (dropout, balanced) in expected.items():
                self.assertEqual(configs[name].channel_dropout, dropout, name)
                self.assertEqual(configs[name].balance_sources, balanced, name)
            # The two `aug` runs are the only ones with augmentation, and they carry the same one.
            augmented = sorted(name for name, cfg in configs.items() if cfg.augment is not None)
            self.assertEqual(augmented, [
                "upernet_inj_ft_balanced_aug_ours", "upernet_inj_ft_balanced_dropsmi_aug_ours",
            ])
            for name in augmented:
                augment = configs[name].augment
                self.assertEqual(
                    (augment.hflip, augment.vflip, augment.rotation,
                     augment.zoom_min, augment.zoom_max),
                    (True, True, 10.0, 0.5, 1.5), name,
                )

if __name__ == "__main__":
    unittest.main()
