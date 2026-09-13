# gsplat-test

Test repository for Gaussian Splat model

## Browser upload interface

```bash
.venv/bin/python -m src.upload_server
```

Open **http://localhost:8001/**. Select a video and set steps, frame interval,
maximum width, output folder, brightness, contrast, and sharpness. Output must be
a new folder inside `runs/`. Uploaded videos are retained in `runs/.uploads/`.
The page shows upload progress followed by the latest engine stdout/stderr line.

Clicking **Upload and build** reserves a waiting tab (to avoid popup blockers).
It navigates to **http://localhost:8000/** when the training viewer is ready.
If popups are blocked, use the **Open viewer** link. The viewer remains open after
export; closing a browser tab does not stop the process.

The engine exits after training and export. The upload page then enables another
upload and suggests a fresh output folder automatically; no server restart is
needed between jobs. One training job runs at a time.

The live viewer runs as a separate persistent process on port 8000 and keeps the
last model visible. When the next job reaches training, that same viewer switches
to its previews and resets its camera and bounding box. It remains alive even if
the upload server exits. Stop the `src.live_viewer` process to shut it down.

Preview snapshots and status are exchanged atomically through `runs/.viewer/`;
viewer startup errors are logged in `runs/.viewer/viewer.log`. On the first launch
after upgrading, close any older viewer already using port 8000. The usual CUDA,
COLMAP, and sharpening dependencies are still required. Stopping the upload server
while training terminates its engine process, so wait for `Model saved:` if you
want the exported result.

Use `--port 8002` to change the upload port (8000 is reserved for the viewer).
For another device on the same network, launch with `--host 0.0.0.0` and open
`http://<computer-IP>:8001/`; the viewer link uses that same hostname on port 8000.
