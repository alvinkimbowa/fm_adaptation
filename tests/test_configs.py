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

        def test_patch_channel_configs_are_explicit(self):
            repo = Path(__file__).resolve().parents[1]
            paths = sorted((repo / "configs").glob(
                "dinov3_convnextt_upernet_ft_aug_p512_red_sci*.yaml"
            ))
            configs = [ExperimentConfig.from_yaml(path) for path in paths]
            self.assertEqual([cfg.train_dataset[:10] for cfg in configs],
                             ["Dataset301", "Dataset302", "Dataset304", "Dataset306", "Dataset307",
                              "Dataset308"])
            self.assertTrue(all(cfg.patching.respect_channels for cfg in configs))
            self.assertTrue(all(cfg.run_name.endswith("_red_ours") for cfg in configs))
            self.assertTrue(all(cfg.refines is None for cfg in configs))

        def test_the_unet_config_trains_everything_at_one_rate(self):
            repo = Path(__file__).resolve().parents[1]
            cfg = ExperimentConfig.from_yaml(repo / "configs" / "unet_aug_p512_red_sci311.yaml")
            self.assertEqual((cfg.model_name, cfg.probe_name, cfg.run_name),
                             ("unet", "unet", "unet_aug_p512_red_ours"))
            self.assertTrue(cfg.train_encoder)
            self.assertIsNone(cfg.encoder_learning_rate)
            self.assertEqual(cfg.train_dataset[:10], "Dataset311")

        def test_weight_mapped_runs_differ_from_the_plain_one_only_in_the_loss(self):
            repo = Path(__file__).resolve().parents[1]
            stem = "dinov3_convnextt_upernet_ft_aug_p512_red{}_sci311.yaml"
            plain, mapped, dice = (ExperimentConfig.from_yaml(repo / "configs" / stem.format(tag))
                                   for tag in ("", "_wmap", "_wmapdice"))
            self.assertEqual([(cfg.weight_maps, cfg.weight_maps_dice) for cfg in (plain, mapped, dice)],
                             [(False, False), (True, False), (True, True)])
            self.assertEqual(len({cfg.run_dir for cfg in (plain, mapped, dice)}), 3)
            self.assertEqual({cfg.train_dataset for cfg in (mapped, dice)}, {plain.train_dataset})
            self.assertEqual({cfg.patching for cfg in (mapped, dice)}, {plain.patching})

        def test_the_zscore_unet_reads_the_stored_channel_alone(self):
            repo = Path(__file__).resolve().parents[1]
            plain = ExperimentConfig.from_yaml(repo / "configs" / "unet_aug_p512_red_sci311.yaml")
            cfg = ExperimentConfig.from_yaml(repo / "configs" / "unet_aug_p512_red_zscore_sci311.yaml")
            self.assertEqual((plain.zscore, cfg.zscore), (False, True))
            self.assertEqual(cfg.run_name, "unet_aug_p512_red_zscore_ours")
            self.assertEqual(plain.input_channels, 3)
            if (cfg.raw_data_dir / cfg.train_dataset).exists():
                self.assertEqual(cfg.input_channels, 1)

        def test_a_refining_run_names_a_run_beside_it(self):
            repo = Path(__file__).resolve().parents[1]
            first = ExperimentConfig.from_yaml(
                repo / "configs" / "dinov3_convnextt_upernet_ft_aug_p512_red_sci307.yaml"
            )
            cfg = ExperimentConfig.from_yaml(
                repo / "configs" / "dinov3_convnextt_upernet_ft_aug_p512_red_refine_sci307.yaml"
            )
            self.assertEqual((cfg.refines, cfg.refines_checkpoint, cfg.refines_plane),
                             (first.run_name, "final", 1))
            self.assertEqual(cfg.refined_run_dir, first.run_dir)
            self.assertNotEqual(cfg.run_dir, first.run_dir)

if __name__ == "__main__":
    unittest.main()
