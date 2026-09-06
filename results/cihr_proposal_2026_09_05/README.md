# CIHR proposal — 2026-09-05

Six one-off DRG Python files, two panel renderers, and their two shell launchers reproduce the proposal work.
Final figures live in `figures/lesion/`, `figures/neurites/`, and `figures/drg/`.
Masks, probability maps, and inference overlays retain their DRG data locations.
Code and reproduction metadata are Git-visible; generated figures are ignored.

## Setup

Run from the fm_adaptation repository root using the existing `.venv-mm` environment:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
code=results/cihr_proposal_2026_09_05
panel=/home/ultrai/GAA/spinal_cord_injury/data/raw_data/DRG-panels
```

The red-only RGB input is `$panel/images/drg_composite.png` (1920 × 1110), SHA256
`35f636ff375a5e0cc0132cbb454bb4535138319e019baf31be96a9288ea7d3b3`.
Saved fold-0 configurations and `best.pt` checkpoints remain under `models/dinov3/`.
The original composite is `$panel/../DRG-composite.png`; annotations and registration
remain under `$panel/masks/` and `$panel/registration/`.
`reproduction_inputs/` preserves package versions, the selected polygon, and
registration transforms. Restore the transforms to `$panel/registration/` if needed.
The current fm_adaptation package and checkpoints are required, not bundled here.

## Panels

```bash
bash "$code/plot_panel_qualitative.sh"
bash "$code/plot_lesion_panel_qualitative.sh"
```

These launchers retain the original model, dataset, and rendering settings, calling
`panel_qualitative.py` and `lesion_panel_qualitative.py` beside them. Both renderers
use shared utilities from `fm_adaptation`. Their default outputs are `figures/neurites/` and
`figures/lesion/`. Case/seed controls are in the launchers; pin the original case
when reproducing a specific panel rather than sampling another case.

## Full-image DRG inference and postprocessing

```bash
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-mm/bin/python "$code/drg_inference.py" --input-dir "$panel"
.venv-mm/bin/python "$code/drg_postprocess.py" --input-dir "$panel"
```

Inference defaults to both plain and eight-view TTA. Use `--prediction plain` or
`--prediction tta` for only one. It selects the original ten models from Datasets
203, 301, 302 (seven variants), and 304, plus the individual Dataset306 topology-2px
model. The four ensembles are all-original-ten, 302-only, 304-only, and 302+304;
Dataset306 is excluded from ensembles. Results use the existing flat names in
`preds/`, `preds_ensemble/`, and `preds_TTA/`.

Saved configurations control patching and red-channel handling. Patch probabilities
use Gaussian overlap blending. TTA averages inverse-aligned probabilities over
covered views: four right-angle rotations × optional horizontal mirror. Ensembles
use equal model probability weights, then threshold >0.5 and skeletonize.
PNG masks and requested `raw/` copies are 1px centerlines encoded 0/255; full
probabilities are in `probabilities/*.npz`. Overlays have filename headers with
TTA/ensemble labels and no suffix for plain predictions.

Postprocessing generates joined, joined+extended, and min12 variants directly.
It retains the original joining and extension algorithms and parameters, and
removes 8-connected components smaller than 12 pixels for min12 variants.
Overlays show original centerlines in cyan, joins in yellow, extensions in magenta.

## Masked-region workflow

Replay the saved exclusion without opening a GUI:

```bash
.venv-mm/bin/python "$code/drg_select_region.py" --input-dir "$panel" \
  --region "$code/reproduction_inputs/region.json"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-mm/bin/python "$code/drg_inference.py" --mode masked-region \
  --input-dir "$panel/masked_region"
.venv-mm/bin/python "$code/drg_postprocess.py" --mode masked-region \
  --input-dir "$panel/masked_region"
```

Omit `--region` for interactive selection. Selection writes masked images,
annotations, exclusion, and polygon metadata; changing a region requires rerunning
its inference. Masked inference uses Dataset304 only. Postprocessing produces the
12 combinations of plain/TTA × original/joined/joined+extended × with/without min12,
enforcing exclusion throughout. Existing matching source predictions are preserved.

## DRG figures

```bash
.venv-mm/bin/python "$code/drg_figures.py" --input-dir "$panel"
```

Default output: **`results/cihr_proposal_2026_09_05/figures/drg/`**. This generates
both comparisons, the revised matched arrows, original spacing/alpha, captions,
and the existing `Fig1`/`Fig2` filename aliases. It uses the masked Dataset304
TTA+joined+extended+min12 prediction. Three-pixel dilation is display-only.
`--operation comparisons` retains original arrows; `--operation arrows` generates
only the adaptive-threshold comparison with revised arrows. Default `all` generates
the final pair. `--prediction-dir` overrides the masked directory (which must also
contain `exclusion.png`); `--source-composite` overrides the original figure.

## Output handling

Every DRG entrypoint accepts `--input-dir`, `--output-dir`, and `--overwrite`.
Defaults preserve the original data locations, except final figures as above.
Existing outputs require explicit `--overwrite`; no staging or publish scripts
are needed. To compare with existing results, choose a new output directory.
For postprocessing separate inference outputs, set `--predictions-dir` accordingly.
Postprocessing expects both plain and TTA predictions (plus baseline ensembles in
full-image mode). The two retained algorithm modules are dependencies, not workflow
entrypoints. The cumulative evaluator also imports the retained extension module.
