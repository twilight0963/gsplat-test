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

This interface runs one model session at a time. After saving, stop the upload
server with Ctrl+C and restart it for another session. Stopping the upload server
terminates its model process, so wait for `Model saved:` before stopping if you
want the exported result. The usual CUDA, COLMAP, and sharpening dependencies
are still required.

Use `--port 8002` to change the upload port (8000 is reserved for the viewer).
For another device on the same network, launch with `--host 0.0.0.0` and open
`http://<computer-IP>:8001/`; the viewer link uses that same hostname on port 8000.
