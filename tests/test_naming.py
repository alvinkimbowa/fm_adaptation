import unittest

from fm_adaptation.naming import dataset_tag, describe_run, run_order


class DescribeRunTests(unittest.TestCase):
    """Reading a configuration name, which no list of experiments takes part in."""

    def test_tokens_read_in_the_order_the_name_writes_them(self):
        self.assertEqual(
            describe_run("upernet_inj_ft_balanced_aug_ours"),
            "UperNet + Inj + FT + balanced + aug + ours",
        )

    def test_an_unknown_token_survives_verbatim(self):
        """The point of deriving rather than registering: a run invented this afternoon labels
        itself, and giving its new idea a nicer word later is an improvement, not a prerequisite."""
        self.assertEqual(
            describe_run("upernet_inj_ft_wobble_ours"),
            "UperNet + Inj + FT + wobble + ours",
        )

    def test_a_name_of_only_unknown_tokens_still_reads(self):
        self.assertEqual(describe_run("frobnicate_v2"), "frobnicate + v2")


    def test_a_trunk_that_is_not_the_default_vit_is_named(self):
        self.assertEqual(
            describe_run("convnext_upernet_ft_p512_ours"),
            "ConvNeXt-L + UperNet + FT + 512px patches + ours",
        )

    def test_each_convnext_size_reads_as_itself(self):
        """The four sizes are compared against each other, so the row has to say which one it is."""
        self.assertEqual(describe_run("convnextt_upernet_ours"), "ConvNeXt-T + UperNet + ours")
        self.assertEqual(describe_run("convnextb_upernet_ours"), "ConvNeXt-B + UperNet + ours")

    def test_a_run_on_projected_slices_says_how_many(self):
        self.assertEqual(
            describe_run("convnextt_upernet_ft_aug_p512_red_s3_wmapdice_ours"),
            "ConvNeXt-T + UperNet + FT + aug + 512px patches + red + 3-slice projections"
            " + weight-mapped + Dice + ours",
        )

    def test_the_unet_trained_from_scratch_reads_as_itself(self):
        self.assertEqual(
            describe_run("unet_aug_p512_red_ours"), "UNet + aug + 512px patches + red + ours"
        )
        self.assertEqual(
            describe_run("unet_aug_p512_red_zscore_ours"),
            "UNet + aug + 512px patches + red + z-score + ours",
        )


class RunOrderTests(unittest.TestCase):
    """Where a run sorts, which also has to fall out of the name."""

    def test_a_refining_run_sorts_straight_after_the_run_it_refines(self):
        names = [
            "convnextt_upernet_ft_aug_p512_red_refine_ours",
            "convnextt_upernet_ft_aug_p512_red_ours",
        ]
        self.assertEqual(sorted(names, key=run_order), names[::-1])
        self.assertTrue(describe_run(names[0]).endswith("red + refine + ours"))

    def test_weight_mapped_runs_sort_straight_after_the_run_they_reweight(self):
        names = [
            "convnextt_upernet_ft_aug_p512_red_ours",
            "convnextt_upernet_ft_aug_p512_red_wmap_ours",
            "convnextt_upernet_ft_aug_p512_red_wmapdice_ours",
        ]
        self.assertEqual(sorted(names[::-1], key=run_order), names)
        self.assertTrue(describe_run(names[2]).endswith("red + weight-mapped + Dice + ours"))

    def test_a_run_sorts_before_the_runs_that_extend_it(self):
        names = [
            "upernet_inj_ft_balanced_aug_ours",
            "upernet",
            "upernet_inj_ft_ours",
            "upernet_inj",
            "upernet_inj_ft_balanced_ours",
        ]
        self.assertEqual(sorted(names, key=run_order), [
            "upernet",
            "upernet_inj",
            "upernet_inj_ft_ours",
            "upernet_inj_ft_balanced_ours",
            "upernet_inj_ft_balanced_aug_ours",
        ])

    def test_ours_does_not_move_a_run_below_its_own_variants(self):
        """`ours` ends almost every name, so ranking it would sort each plain run after everything
        built on top of it -- the reverse of what a reader wants down a column."""
        self.assertLess(run_order("upernet_inj_ft_ours"),
                        run_order("upernet_inj_ft_balanced_ours"))

    def test_an_unknown_token_sorts_after_every_known_one(self):
        """A new idea lands in one predictable place rather than displacing the runs around it."""
        self.assertLess(run_order("upernet_inj_ft_balanced_ours"),
                        run_order("upernet_inj_ft_wobble_ours"))

    def test_sweep_suffixes_stay_with_the_run_they_vary_from(self):
        names = ["linear_finetune_wd10.0", "linear_finetune", "linear_finetune_wd1.0", "linear"]
        self.assertEqual(sorted(names, key=run_order),
                         ["linear", "linear_finetune", "linear_finetune_wd1.0",
                          "linear_finetune_wd10.0"])


class DatasetTagTests(unittest.TestCase):
    def test_the_number_and_the_shared_stain_suffix_are_dropped(self):
        self.assertEqual(dataset_tag("Dataset219_lesion_MYK_smi_gfap"), "lesion_MYK")

    def test_a_name_without_either_is_left_alone(self):
        self.assertEqual(dataset_tag("BUSBRA_GE_Logiq_5"), "BUSBRA_GE_Logiq_5")


if __name__ == "__main__":
    unittest.main()
