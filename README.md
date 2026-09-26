# gsplat-test

Test repository for Gaussian Splat model

## Browser upload interface

```bash
.venv/bin/python -m src.upload_server
```

Open **http://localhost:8001/**. Select a video and set steps, frame interval,
maximum width, output folder, brightness, contrast, and sharpness. Output must be
a new folder inside `runs/`. Uploads are kept in `runs/.uploads/` while a job runs.
The page shows upload progress followed by the latest engine stdout/stderr line.

Photo datasets can be uploaded too. Drag the folder that directly contains the photos
(e.g. `Dataset/ATR/Images`) anywhere onto the page, or set **Input type** to *Photo
folder* and use **Choose folder…** or **Or select photos** (for phones). Only
JPEG/PNG/TIFF files directly inside the folder are sent; subfolders and other files
(`Thumbs.db`, metadata text, ground-control CSVs) are ignored, so dropping
`Dataset/ATR` itself finds no photos. Dropping a single video selects it as the input.
The page shows how many photos will be uploaded before you start. Files are sent
unchanged, so EXIF GPS and DJI calibration reach the engine. **Use every Nth photo**
maps to `--photo-every` (default 2). If the browser disconnects mid-upload (page
reloaded or closed), the server logs one line, discards the partial upload and is
ready for the next one.

When an uploaded job saves its model, the server deletes the upload from
`runs/.uploads/` and the run's `capture/`, `capture_subset/`, `colmap/` and `sr/`
folders. It keeps `model.glb`, `benchmark.json`, `benchmark.log` and
`photo_metadata.json`. Failed jobs keep everything for inspection or a retry.
Command-line runs are not cleaned up, so their COLMAP cache can be reused.

Clicking **Upload and build** keeps you on the upload page. A new tab opens at
**http://localhost:8000/** only when the training viewer reports it is ready.
Browsers may block delayed popups; if so, use the **Open viewer** link. The viewer remains open after
export; closing a browser tab does not stop the process.

The engine exits after training and export. The upload page then enables another
upload and suggests a fresh output folder automatically; no server restart is
needed between jobs. One training job runs at a time.

**Stopping a job early.** The upload page's Progress panel and the viewer's top bar
both offer two controls while a job runs:

- **Stop & save** (during training) ends training after the current step and
  exports the model trained so far. The benchmark records the steps actually run
  (`training_steps_completed`), and the status reads "stopped early".
- **Discard** (any time after the upload; click twice to confirm) ends the job
  without saving and deletes its upload and output folder. Before training starts
  there is no model yet, so only Discard is offered. **Cancel upload** aborts an
  upload in progress.

Stop requests go through `runs/.viewer/stop.json` and name the running session, so a
request can never stop a later job. The engine exits with code 3 when discarded.

**Viewer controls.** Drag to orbit or pan (chosen in the toolbar); right-drag,
middle-drag or Shift-drag always pans. Scroll or pinch to zoom; two fingers also
pan on touch screens. Zoom and roll buttons repeat while held. Press **H** in the
viewer for keyboard shortcuts (arrows, +/−, O/P, Z/C/X, U, R). The edge splat size
limit is under **Display**. The viewer is a persistent process; when a job starts
and finds a viewer running older code, it replaces it automatically (viewers from
before this feature must be restarted once by hand).

The live viewer runs as a separate persistent process on port 8000 and keeps the
last model visible. When the next job reaches training, that same viewer switches
to its previews and resets its camera and bounding box. It remains alive even if
the upload server exits. Stop the `src.live_viewer` process to shut it down.

The bounding box keeps the whole reconstruction in view. It is fitted around every
region connected to the scene (weighted by opacity), so thin structures joined to
it, such as towers and masts, stay inside while isolated floaters are left out; it
follows the scene's main axes and is not forced into a cube, so wide drone sites fit
without empty space. The same box is used for live previews (from the COLMAP
points) and the final model, and the viewer refits saved models when it opens them,
so models made with the older, tighter box also open in full. The starting camera
looks down at about 32 degrees and backs off only as far as needed for the whole box
to fit the window; **Reset** returns to that view.

Preview snapshots and status are exchanged atomically through `runs/.viewer/`;
viewer startup errors are logged in `runs/.viewer/viewer.log`. On the first launch
after upgrading, close any older viewer already using port 8000. The usual CUDA,
COLMAP, and sharpening dependencies are still required. Stopping the upload server
while training terminates its engine process, so wait for `Model saved:` if you
want the exported result.

Use `--port 8002` to change the upload port (8000 is reserved for the viewer).
For another device on the same network, launch with `--host 0.0.0.0` and open
`http://<computer-IP>:8001/`; the viewer link uses that same hostname on port 8000.


### Video keyframes and runtime validation

`--every 10` first samples the source video. Adaptive selection then operates on
those candidates, using tracked image displacement and sharpness. It retains
endpoints, keeps both sides of tracking breaks, and limits candidate gaps to six
by default. Tracking failure retains extra views; there is no hard frame-count
cap or guaranteed runtime. Image motion is a coverage heuristic, not metric
parallax or proof of reconstruction accuracy. Photo selection is unchanged.

```bash
.venv/bin/python -m src.engine INPUT.mp4 --output runs/keyframes \
  --every 10 --max-width 960 --steps 2000 --headless
```

Use `--keyframe-max-gap 1` to disable adaptive selection. Adjust
`--keyframe-max-gap` (default 6), `--keyframe-motion` (default 0.06 of the image
diagonal), and `--sequential-overlap` (default 12) for different capture motion.
`--vocab-tree vocab_tree.bin` enables COLMAP retrieval-based loop detection;
retaining revisited frames alone does not ensure loop closure. The original
video matching overlap can be restored with `--sequential-overlap 30`.

Selected frames and their candidate indices/reasons are recorded in
`capture/keyframes/<input-and-settings-hash>/keyframes.json`. Original candidates
are preserved. Benchmark reports distinguish `keyframe_candidates` from
`frames_before_colmap`, report keyframe-selection time and individual COLMAP
command subtotals, and state whether the run met the runtime target: 1.5x the
input video's duration (e.g. a 10-minute video within 15 minutes, whatever `--every`
is), or 600 s for photo sets. Command subtotals are included in COLMAP time and must
not be added again. Every run also reports the final model's PSNR/SSIM on every 8th
of the views it was trained on (`model_psnr`, `model_ssim` under `quality`); this
shows how well the model fits its inputs, while `--eval-every` measures held-out views.

Video matching uses a dense window (`quadratic_overlap=0`): COLMAP's quadratic
mode replaces the window with gaps 1, 2, 4, 8, ..., which verified about half as
many pairs and made `global_mapper` about twice as slow on test footage.
`--mapper-tracks-per-view` (default 1000, `0` = keep all) limits the tracks
`global_mapper` optimizes; on a 319-view test it cut mapping time by ~25% with
the same registration, reprojection error and held-out PSNR.

### Training quality

Training uses L1 plus a D-SSIM term (`--ssim-weight`, default 0.2; `0` restores
pure L1). With the optional fused CUDA SSIM kernel installed, the term runs at full
resolution in ~3 ms per step; without it, it falls back to a PyTorch SSIM at half
resolution (~7 ms). Install the kernel into the venv (needs the CUDA toolkit):

```bash
CUDA_HOME=/opt/cuda .venv/bin/python -m pip install --no-build-isolation \
  "git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
```

Training also scales the position learning
rate by the camera extent, as in 3DGS (`--no-scale-means-lr` disables it). Use
`--eval-every 8` to hold out every 8th registered view and report PSNR/SSIM in
`benchmark.log`/`benchmark.json` under `quality`; held-out views are not trained on,
so leave it off for production models.

Densification defaults to 3DGS-MCMC (`--densify mcmc`, gsplat's `MCMCStrategy`):
near-transparent Gaussians are relocated to where opacity is high and the count
grows 5% every `--refine-every` steps (default 50) up to `--max-gaussians`
(default 1,000,000), stopping at 80% of training. COLMAP points alone are far too
few (a single training view could only reach ~24 dB PSNR; with MCMC it reaches
~39 dB). On a 1800-frame drone clip (66 keyframes, 1280 px, held out every 8th
view), 5000 steps (the default) gave:

| Densification | Training | Held-out PSNR | Gaussians |
|---|---|---|---|
| voxel (previous default) | 185 s | 19.8 dB | 20k |
| MCMC, refine every 100 | 246 s | 22.3 dB | 126k |
| MCMC, refine every 50 (default) | 461 s | 24.4 dB | 697k |

Training time grows with image size; 1920 px frames have 2.25x the pixels of 1280 px,
so expect roughly twice the training time (estimate, not measured).
`--densify voxel` restores the earlier DroneSplat-style voxel growth. MCMC needs CUDA.

Absolute geographic accuracy is reported as **unverified**. Photo EXIF GPS is
used for neighbor matching, not an implemented geographic alignment or checkpoint
evaluation. Video-only inputs have no geographic reference. A runtime pass is
not a claim of <=1 m absolute error.

### Photo directories and metadata

Every completed engine run writes `benchmark.log` (readable summary) and
`benchmark.json` (machine-readable timings) in its output directory. Reports
include UTC start/end timestamps, total wall time, decoded video frames,
selected/imported frames before COLMAP, usable frames from the selected COLMAP
reconstruction, and separate durations for input preparation, preprocessing,
CAS, SR, COLMAP, reconstruction loading, viewer startup, training and export.
CAS/SR are not double-counted in other stages; preprocessing total is their
subtotal plus brightness/contrast, unsharp and target preparation. COLMAP time
includes undistortion. Training time includes target loading, preview publishing
and final box fitting. Failed runs do not generate a completed-run report.

The engine also accepts a directory of JPEG, PNG or TIFF photographs:

```bash
.venv/bin/python -m src.engine Dataset/Images --output runs/tower-photos --max-width 1920
```

All photos directly inside the directory are used; `--every` applies only to
video. Use a new output directory and photos with consistent dimensions and
upright EXIF orientation. Photos wider than `--max-width` are written once at that
width into `capture/photos` (Lanczos, JPEG quality 95), keeping EXIF (GPS, focal
length); otherwise the originals are copied byte-for-byte. Staging once means COLMAP
never decodes full-size originals: on 305 8192 px photos, feature extraction went
from 88 s to 7 s and undistortion from 124 s to 3 s. `photo_metadata.json` records
the source-to-capture mapping and scale. Files are ordered by capture time when every image has
a timestamp, otherwise by filename.

When all photos have valid EXIF GPS, auto matching combines 20 GPS neighbors per
image (100 m search radius, horizontal distance) with sequential overlap 10.
Change the neighbor budget with `--spatial-neighbors`. With incomplete GPS,
matching falls back to exhaustive. `--image-matching exhaustive` explicitly
selects all-pairs matching for a quality comparison. GPS selects candidate pairs;
it is not treated as exact geometry, and gimbal angles are not imposed as poses.

Consistent DJI calibrated focal length and optical center seed a shared
SIMPLE_RADIAL camera, with lens distortion and focal length still refined by
COLMAP. If calibration is missing or differs, COLMAP estimates the appropriate
parameters instead. Metadata is read from the originals; the DJI calibration is
scaled to the staged size, so camera coordinates stay consistent. For portrait photos the equivalent longest-side limit is used.
Brightness/contrast, unsharp masking and CAS run on the undistorted training
targets. `--super-resolution` also works with photo directories; its 2x targets
and base-image loss use the same undistorted cameras.

This optimization reduces the candidate matching workload; end-to-end runtime
and reconstruction completeness still depend on overlap and scene content.

### Optional SwinIR super-resolution training

`src.engine --super-resolution` enables an SRGS-inspired 2× pipeline. COLMAP
uses unenhanced extracted frames. After undistortion, brightness/contrast produce
lossless base targets; SwinIR-S creates enhanced targets, then CAS sharpens them.
Training blends L1 against enhanced targets with L1 between area-downsampled
renders and base targets (`--sr-prior-weight`, default 0.5). This is an adaptation,
not a reproduction of the paper's full training recipe or a guarantee of mapping
accuracy. Camera intrinsics are doubled with image dimensions.

Install the optional dependency and obtain the official architecture and weights:

```bash
.venv/bin/python -m pip install -r requirements-sr.txt
git clone https://github.com/JingyunLiang/SwinIR.git third_party/SwinIR
mkdir -p weights
curl -fL https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth -o weights/swinir-lightweight-x2.pth
```

Use a new output directory for each run (COLMAP databases are not interchangeable
between preprocessing modes). Run from the project root:

```bash
.venv/bin/python -m src.engine input.mp4 --output runs/sr-example \
  --super-resolution \
  --max-width 640 --sr-tile 128 --sharpness 0.2 --steps 5000
```

The engine CLI uses the same default SwinIR source and checkpoint locations as
the upload server (`third_party/SwinIR` and `weights/swinir-lightweight-x2.pth`).
The path flags are only needed for a custom installation.

SR defaults to one camera per training step, with unsharp masking disabled.
`--unsharp` explicitly enables it after SR, before CAS; `--no-unsharp` disables
it in the existing mode too. Brightness remains an additive pixel offset and
contrast a multiplier. CAS remains enabled, including at sharpness 0 (minimum
extra sharpening, not a bypass). SR inference runs overlapping tiles in batches
of 8 in fp16 on the GPU (about 2x faster than one fp32 tile at a time, with the
same output to within 65 dB PSNR), in a separate process that exits before
training begins. CAS for SR targets runs on the GPU in linear light, matching
FidelityFX_CLI to within about 0.2-0.45/255 on average. SR targets are held in
RAM during training when they fit in half of the available memory (about 16 MB
per view at 1280 px input); otherwise each view is read from disk a few steps
ahead of training. The existing Gaussian
budget controls still apply. These defaults are a starting point for 6 GB GPUs;
actual peak memory depends on scene complexity and the live viewer as well.

Without `--super-resolution`, the enhancement mode uses batch size 4 by default.
For both videos and photo directories, COLMAP receives unenhanced images;
brightness/contrast, unsharp and CAS are applied afterward to registered,
undistorted training images only. Video extraction still applies `--every` and
`--max-width` before COLMAP. Use a fresh output directory when changing pipelines.
The web upload interface includes the SR mode, tile size, enhanced-image weight,
and automatic/on/off unsharp controls. After the setup above, restart
`python -m src.upload_server` and select **On — SwinIR 2× + CAS**. Set maximum
width to 640 and CAS to 0.2 as an initial 6 GB configuration. The existing
brightness/contrast controls apply in both modes. The viewer still opens only
when splat training starts, after SR preprocessing finishes.

For custom installation locations, launch the upload server with
`--swinir-root /path/to/SwinIR --sr-checkpoint /path/to/weights.pth`.
These paths refer to the server machine, not the uploading phone/computer.
SwinIR's source and weights are external prerequisites; they are not downloaded
during training. Missing source/weights are reported before accepting the job.

Jobs launched by `src.upload_server` pass `--use-server`, selecting the
persistent browser viewer. Direct `src.engine` runs default to the desktop
OpenCV viewer; use `--use-server` explicitly when a browser viewer is desired.
