"""Slabs of a z-stack, for datasets that store one case per slice.

A slab is a slice together with its neighbours in depth. What is read for it -- an image, a label,
a weight map -- is the per-pixel maximum over its slices, which is the projection a microscope user
would make of that part of the stack.
"""

import re

# A case that is one slice of a stack ends in its one-based position: `..._z042`.
SLICE_ID = re.compile(r"(.*_z)(\d{3})$")


def slab_members(case_ids, slices):
    """Each case mapped to the cases its slab is made of, itself included, in depth order.

    The slab of slice `z` runs from `z - (slices - 1) // 2` to `z + slices // 2` and is cut short at
    the ends of the stack, so the first slice of a 3-slice slab has two members. Only cases named in
    `case_ids` can be members. A case that is not a slice of a stack is its own slab.
    """
    known = set(case_ids)
    members = {}
    for case_id in case_ids:
        match = SLICE_ID.match(case_id)
        if slices <= 1 or match is None:
            members[case_id] = (case_id,)
            continue
        stem, centre = match.group(1), int(match.group(2))
        window = range(centre - (slices - 1) // 2, centre + slices // 2 + 1)
        members[case_id] = tuple(
            name for name in (f"{stem}{position:03d}" for position in window) if name in known
        )
    return members
