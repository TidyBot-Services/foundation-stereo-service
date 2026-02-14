# FoundationStereo Depth Estimation Service

> TidyBot backend service for zero-shot stereo depth estimation using [FoundationStereo](https://github.com/NVlabs/FoundationStereo) (CVPR 2025 Best Paper Nomination).

Accepts a pair of rectified stereo images and returns dense disparity and/or metric depth maps. Generalizes zero-shot across diverse scenes without per-dataset fine-tuning — handles reflective, transparent, and textureless surfaces where traditional IR stereo (e.g., RealSense) fails.

## Quick Start

```bash
# Start the service
cd /home/qifei/foundation-stereo-service
python main.py
# Runs on http://0.0.0.0:8001
```

## API Reference

### `GET /health`

Check service health and GPU status.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "gpu_memory_mb": 32768,
  "model_loaded": true,
  "checkpoint": ".../pretrained_models/23-51-11/model_best_bp2.pth",
  "valid_iters": 32
}
```

### `POST /depth`

Run stereo depth estimation on a rectified image pair.

**Request Body:**
```json
{
  "left_image": "<base64 encoded JPEG/PNG>",
  "right_image": "<base64 encoded JPEG/PNG>",
  "focal_length": 382.5,
  "baseline": 0.055,
  "valid_iters": 32,
  "scale": 1.0,
  "hiera": false,
  "return_disparity": true,
  "return_depth": true,
  "return_vis": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `left_image` | string | ✅ | — | Base64-encoded left rectified image |
| `right_image` | string | ✅ | — | Base64-encoded right rectified image |
| `focal_length` | float | ✅ | — | Focal length in pixels (fx) |
| `baseline` | float | ✅ | — | Stereo baseline in meters |
| `valid_iters` | int | ❌ | 32 | GRU iterations (higher = better, slower) |
| `scale` | float | ❌ | 1.0 | Downscale factor (≤1.0) |
| `hiera` | bool | ❌ | false | Hierarchical mode for >1K images |
| `return_disparity` | bool | ❌ | true | Return disparity map |
| `return_depth` | bool | ❌ | true | Return metric depth map |
| `return_vis` | bool | ❌ | false | Return colorized visualization |

**Response:**
```json
{
  "disparity": "<base64 numpy float32 array HxW>",
  "depth": "<base64 numpy float32 array HxW>",
  "vis": "<base64 PNG image>",
  "height": 480,
  "width": 640,
  "inference_ms": 245.3,
  "device": "cuda"
}
```

- **disparity**: Dense disparity in pixels (float32). Decode with `np.load(io.BytesIO(base64.b64decode(data)))`.
- **depth**: Metric depth in meters aligned to the left camera frame. Computed as `focal_length * baseline / disparity`.

## Python Client SDK

```python
from client import FoundationStereoClient

client = FoundationStereoClient("http://192.168.1.100:8001")

# Basic usage
result = client.depth(
    left_image="left.png",
    right_image="right.png",
    focal_length=382.5,
    baseline=0.055,
)
depth_map = result["depth"]        # np.ndarray (HxW) in meters
disparity = result["disparity"]    # np.ndarray (HxW) in pixels

# RealSense convenience method
result = client.depth_from_realsense("left_ir.png", "right_ir.png")

# With numpy arrays (e.g., from RealSense SDK)
result = client.depth(
    left_image=left_np,    # HxWx3 RGB numpy array
    right_image=right_np,
    focal_length=fx,
    baseline=0.055,
)

# Get visualization
result = client.depth("left.png", "right.png", focal_length=382.5, baseline=0.055, return_vis=True)
with open("vis.png", "wb") as f:
    f.write(result["vis"])
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FS_CKPT_DIR` | `vendor_fs/pretrained_models/23-51-11/model_best_bp2.pth` | Path to model checkpoint |
| `FS_VALID_ITERS` | `32` | Default GRU iterations |

## Model

Uses FoundationStereo ViT-Large (23-51-11), the best-performing model from the paper. Ranked #1 on both Middlebury and ETH3D stereo benchmarks.

- **Paper**: [FoundationStereo: Zero-Shot Stereo Matching](https://arxiv.org/abs/2501.09898)
- **Authors**: Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield (NVIDIA)

## Requirements

- Python 3.11+
- CUDA GPU with ≥16GB VRAM (24GB+ recommended)
- PyTorch 2.4+
- See `requirements.txt`
