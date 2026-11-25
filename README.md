# FastMOT

FastMOT tracks pedestrians from an IP camera on a CPU and shows live bounding boxes and track IDs in a browser. It also processes recorded video from the command line.

[Quick start](#quick-start) · [Architecture](#architecture) · [Deployment](deploy/camera/README.md)

## Tracking demo

<img src="deploy/video/tracking-demo.gif" width="720" alt="Pedestrians with colored bounding boxes and track IDs">

## Architecture

<img src="deploy/architecture.svg" width="720" alt="RTSP camera architecture with 4 FPS tracking and 30 FPS optical-flow display targets, delivered to a LAN browser through Caddy">

## Quick start

### CPU setup

From the repository root on Linux with Python 3.11:

```bash
python -m venv .runtime/venv
source .runtime/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Connect an IP camera

Save the camera's RTSP URL in `.runtime/camera/rtsp-url` with file permissions `0600`. Use a camera account with viewing access. Follow the [RTSP credential setup](deploy/camera/README.md#1-chạy-với-camera-ip) for the commands.

```bash
python scripts/web_tools/serve_camera.py --rtsp-url-file .runtime/camera/rtsp-url
```

### Simulate a camera

Install FFmpeg and put MediaMTX on your `PATH`. The script publishes the included clip over localhost RTSP and tests reconnection.

```bash
python scripts/web_tools/verify_camera.py \
  --mediamtx "$(command -v mediamtx)" \
  --fixture deploy/video/fixture.mp4 \
  --seconds 60 --output outputs/camera_sim
```

## Process a recorded video

```bash
python scripts/inference/run_tracking.py \
  --source path/to/video.mp4 \
  --model-config configs/models/yolo26s_1280_openvino_fp32.json \
  --output outputs/runs/my_video
```

## License

FastMOT source code is licensed under the GNU Affero General Public License v3.0 only (AGPL-3.0-only). [Full license text](LICENSE)

The bundled YOLO26s detector and YOLO26n ReID weights originate from Ultralytics models licensed under AGPL-3.0. [Ultralytics licensing](https://www.ultralytics.com/license)

The simulation fixture and tracking GIF are based on the MOT17-04-FRCNN sequence from MOTChallenge and are shared under CC BY-NC-SA 3.0. The fixture encodes the source frames as video, and the GIF adds FastMOT tracking overlays. [MOT17 source and license](https://motchallenge.net/data/MOT17/) · [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/)
