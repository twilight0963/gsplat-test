"""Show a saved .glb model in the persistent browser viewer (the upload page's Open model)."""
import argparse
import struct
import sys
from pathlib import Path

from src.gltf_gsplat import read_gsplat_glb
from src.live_viewer import PreviewPublisher, ensure_viewer

VIEWER_DIR = Path(__file__).resolve().parent.parent / 'runs' / '.viewer'


def main():
    parser = argparse.ArgumentParser(description='Open a Gaussian splat .glb in the browser viewer on port 8000.')
    parser.add_argument('model', type=Path, help='A .glb written by src.engine (KHR_gaussian_splatting).')
    parser.add_argument('--name', help='Name shown in the viewer (default: the file name).')
    parser.add_argument('--width', type=int, default=1920, help='Viewer render width if it has to start (default: 1920).')
    parser.add_argument('--height', type=int, default=1080, help='Viewer render height if it has to start (default: 1080).')
    args = parser.parse_args()
    name = args.name or args.model.name
    try:
        data = read_gsplat_glb(args.model)
    except (OSError, ValueError, KeyError, IndexError, TypeError, struct.error) as exc:
        # json.JSONDecodeError and UnicodeDecodeError are ValueErrors. Name the upload, not its temporary path.
        print(f'{name} is not a Gaussian splat model this viewer can open: '
              f'{str(exc).replace(str(args.model), name)}', flush=True)
        sys.exit(2)
    ensure_viewer(VIEWER_DIR, args.width, args.height)
    publisher = PreviewPublisher(VIEWER_DIR)
    # Not a training run: mark it finished first, so the viewer never offers to stop it.
    publisher.finish(message=f'Opening {name}')
    publisher.publish(data, 1, 1)
    publisher.finish(message=f'Saved model: {name}')
    print(f'Model opened: {name} ({len(data["means"]):,} splats)', flush=True)


if __name__ == '__main__':
    main()
