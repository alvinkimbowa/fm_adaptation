import unittest

from fm_adaptation.slabs import slab_members


class SlabMemberTests(unittest.TestCase):
    """Which slices of a stack are read together."""

    def setUp(self):
        self.stack = [f"Src__a_z{z:03d}" for z in range(1, 6)]

    def test_a_slice_is_read_with_its_neighbours(self):
        members = slab_members(self.stack, 3)
        self.assertEqual(members["Src__a_z003"], ("Src__a_z002", "Src__a_z003", "Src__a_z004"))

    def test_the_slab_is_cut_short_at_the_ends_of_the_stack(self):
        members = slab_members(self.stack, 3)
        self.assertEqual(members["Src__a_z001"], ("Src__a_z001", "Src__a_z002"))
        self.assertEqual(members["Src__a_z005"], ("Src__a_z004", "Src__a_z005"))

    def test_an_even_slab_takes_its_extra_slice_from_above(self):
        self.assertEqual(slab_members(self.stack, 2)["Src__a_z003"], ("Src__a_z003", "Src__a_z004"))

    def test_stacks_do_not_lend_each_other_slices(self):
        members = slab_members(self.stack + ["Src__b_z002", "Src__b_z003"], 3)
        self.assertEqual(members["Src__b_z002"], ("Src__b_z002", "Src__b_z003"))

    def test_one_slice_and_cases_that_are_not_slices_stand_alone(self):
        self.assertEqual(slab_members(self.stack, 1)["Src__a_z003"], ("Src__a_z003",))
        windows = ["Src__a_z001-010", "Src__a_z006-015", "Src__plain"]
        self.assertEqual(slab_members(windows, 3), {name: (name,) for name in windows})


if __name__ == "__main__":
    unittest.main()
