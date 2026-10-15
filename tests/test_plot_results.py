import unittest

from fm_adaptation.plot_results import ALL, group_cases


class GroupCasesTests(unittest.TestCase):
    values = {"a": ("wrist",), "b": ("elbow",), "c": ("distal forearm",), "d": (),
              "e": ("wrist", "elbow")}

    def test_groups_follow_the_order_given_then_all(self):
        groups = group_cases(list(self.values), self.values, ["wrist", "elbow"])
        self.assertEqual(list(groups), ["wrist", "elbow", ALL])
        self.assertEqual(groups["wrist"], ["a", "e"])

    def test_all_leaves_out_groups_not_shown_but_keeps_cases_with_no_value(self):
        groups = group_cases(list(self.values), self.values, ["wrist", "elbow"])
        self.assertEqual(groups[ALL], ["a", "b", "d", "e"])

    def test_an_empty_order_keeps_every_value_sorted(self):
        groups = group_cases(list(self.values), self.values, [])
        self.assertEqual(list(groups), ["distal forearm", "elbow", "wrist", ALL])
        self.assertEqual(len(groups[ALL]), 5)


if __name__ == "__main__":
    unittest.main()
