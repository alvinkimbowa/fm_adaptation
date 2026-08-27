"""Turning the names on disk into something a person can read.

A run is a directory: `<model>/<trained on>/<configuration>/fold_<n>`. The configuration is a name
built out of `_`-separated tokens, and every part of this project that shows a run to someone -- a
results table, a figure column header -- needs the same two answers: what does this name say, and
where does this run sort against its neighbours.

Both answers come from the name itself. Nothing here lists experiments: adding a run, or deleting
one, needs no edit. `TOKENS` grows only when a genuinely new idea appears, and an idea it has never
heard of still reads and still sorts -- just in its raw form until someone gives it a word.
"""

import re

# Model families, which is what these are -- not experiments. New ones appear on the timescale of
# adopting a new foundation model.
MODEL_NAMES = {
    "sam3": "SAM3",
    "dinov3": "DINOv3",
    "nnunet": "nnU-Net",
}


# The vocabulary run names are built from, in the order a name accretes them: what the probe is,
# then what was done to it. Order matters twice over -- it is the display order within a name and
# the sort order between names -- so a token belongs where its idea belongs, not alphabetically.
TOKENS = (
    # The probe or decoder.
    ("linear", "LP"),
    ("nonlinear", "NLP"),
    ("upernet", "Adapter + UperNet"),
    ("m2f", "Adapter + Mask2Former"),
    # What was added to it.
    ("inj", "Inj"),
    ("kd", "KD"),
    ("ft", "FT"),
    ("finetune", "FT"),
    # Trunk size, where it is not the default ViT-L.
    ("vitb", "ViT-B"),
    ("vits", "ViT-S"),
    # Schedule and initialisation.
    ("poly", "poly"),
    ("init", "init"),
    # How the training set was sampled and perturbed.
    ("balanced", "balanced"),
    ("aug", "aug"),
    ("dropsmi", "drop-SMI"),
    ("dropany", "drop-any"),
    ("gfap", "GFAP-only"),
    # Marks a configuration as this project's own rather than a reproduction, and so comes last.
    ("ours", "ours"),
)

# Read but not ranked. `ours` says where a configuration came from rather than what was done to it,
# and it sits at the end of almost every name -- ordering on it would push each plain run below the
# variants that extend it, which is backwards. Dropping it lets a name sort as a prefix of the names
# that build on it, which is exactly the relationship worth reading down a column.
UNRANKED = frozenset({"ours"})

_DISPLAY = dict(TOKENS)
_POSITION = {token: index for index, (token, _) in enumerate(TOKENS)}


def _tokens(run_name):
    """The words a name is built from. `__` is a separator too, so it yields no empty token."""
    return [token for token in run_name.split("_") if token]


def describe_run(run_name):
    """A configuration name as prose: `upernet_inj_ft_balanced_aug_ours` -> the words it is made of.

    An unknown token is passed through as it was written. That is deliberate: a new experiment is
    readable the moment it exists, and giving it a nicer word is an improvement rather than a
    prerequisite.
    """
    if not run_name:
        return ""
    return " + ".join(_DISPLAY.get(token, token) for token in _tokens(run_name))


def run_order(run_name):
    """Sort key placing a run beside the run it varies from.

    These names grow by accretion -- `upernet`, then `upernet_inj`, then `upernet_inj_ft_ours` --
    so ordering on the vocabulary positions of the tokens, in sequence, reproduces the family tree
    without anyone maintaining a ranking. A token the vocabulary does not know sorts after every
    token it does, by its own text, so a new name lands in one predictable place instead of moving
    the runs around it.
    """
    if not run_name:
        return ()
    return tuple(
        (_POSITION[token], "") if token in _POSITION else (len(TOKENS), token)
        for token in _tokens(run_name)
        if token not in UNRANKED
    )


def dataset_tag(dataset):
    """A dataset directory as a short name: the number and the stain suffix carry no information
    once several of these sit side by side, and every SCI set ends in the same one."""
    return re.sub(r"_(smi_)?gfap$", "", re.sub(r"^Dataset\d+_", "", dataset))
