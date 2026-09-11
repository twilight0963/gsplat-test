# Splat quality investigation

The reported symptoms are angle-dependent transparency, fog-like floaters, and
stretched lines. Changes apply to new training runs, not existing GLB files.

## Evidence

Inspection of seven saved models found 18–58% of splats with longest/shortest
axis ratios above 10. For example, `runs/building-sharp/model.glb` contains
135,406 splats, 59,361 with ratios above 10, and 845 with opacity below 0.005.
Anisotropy alone does not prove an artifact: thin surfaces also need anisotropic
splats. No splats were removed from these saved models.

Code inspection found:

- Voxel pruning deleted opaque singleton voxels solely for having low occupancy.
- Voxel-average opacity could hide individually near-invisible splats.
- Damping all gradients could freeze the scale and opacity of rogue splats.
- Growth produced non-leaf optimizer tensors and could exceed the point budget.
- Opacity resets increased already faint splats to 0.01 and allowed pruning
  before a full pass through the views could recover opacity.
- Sampling discarded the final partial batch of each shuffled set of views.
- The single most isolated initial point controlled the global scale ceiling.

## Changes and tradeoffs

Preserve opaque singleton voxels; prune sparse voxels only when their average
opacity is also low. Independently remove splats below 0.005 opacity. Skip a
pruning event if it would erase the entire model. This guard preserves the
ability to recover, but deliberately retains fog if every splat is weak.

Apply a soft axis-ratio penalty above 10, weighted by 0.01, at each step. This
follows the scale-ratio regularization approach documented by Nerfstudio, with
our own conservative weight and update frequency. It is a heuristic, not a
scene-validated optimum. `--scale-regularization 0` disables it;
`--max-axis-ratio` adjusts the threshold. Stronger regularization may damage
thin geometry. Scales and opacities remain free to optimize under voxel damping.

Use the 95th percentile of initial neighbor distances, multiplied by 20, for
the scale ceiling instead of the maximum. This limits the influence of isolated
points but can constrain genuinely sparse regions. Preserve optimizer leaves
and state during growth, and honor the remaining growth budget.

Reset opacities by capping them rather than raising faint splats. Pause refinement
for at least one pass through the training views (and at least one pruning
interval), and avoid resets too close to completion. Include partial view batches.

## Research sources

- [Nerfstudio quality and regularization](https://docs.nerf.studio/nerfology/methods/splat.html#quality-and-regularization)
  identifies spiky Gaussians and describes scale regularization; it also notes
  that less aggressive opacity culling can improve quality.
- [Nerfstudio implementation](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py)
  implements a soft longest/shortest-axis ratio penalty.
- [gsplat default strategy](https://docs.gsplat.studio/main/_modules/gsplat/strategy/default.html)
  uses per-Gaussian opacity pruning and provides a pause after opacity resets,
  suggesting a duration related to the training-image count.

## Validation and limits

Run `.venv/bin/python -B -m unittest discover -s tests -v`.
The CPU tests cover pruning, gradient recovery, growth/Adam state, opacity resets,
regularizer gradient direction, and the existing training-preview functionality.
GPU rasterization is mocked in preview tests. CUDA was unavailable for this
investigation, so no before/after rendering quality or speed claim is made.

For visual acceptance, retrain into a fresh output directory to avoid reusing
an old COLMAP database. Compare the same cameras, including the problematic
angles, and check that reduced fog does not come with missing thin surfaces.
Angle-dependent transparency can also reflect missing camera coverage or
insufficient geometry; these changes do not reconstruct unseen surfaces.
