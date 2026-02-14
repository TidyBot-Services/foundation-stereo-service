"""
FoundationStereo Depth Estimation Service — TidyBot Backend
Hosted on FastAPI. Accepts stereo image pairs and returns disparity + metric depth maps.
"""

import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# Add vendor FoundationStereo to path
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_fs")
sys.path.insert(0, VENDOR_DIR)

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from core.foundation_stereo import FoundationStereo

# ─── Globals ──────────────────────────────────────────────────────
MODEL = None
MODEL_CFG = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.environ.get(
    "FS_CKPT_DIR",
    os.path.join(VENDOR_DIR, "pretrained_models", "23-51-11", "model_best_bp2.pth"),
)
VALID_ITERS = int(os.environ.get("FS_VALID_ITERS", "32"))


def load_model():
    global MODEL, MODEL_CFG
    cfg_path = os.path.join(os.path.dirname(CKPT_DIR), "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = VALID_ITERS
    MODEL_CFG = cfg

    model = FoundationStereo(cfg)
    ckpt = torch.load(CKPT_DIR, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()
    MODEL = model
    print(f"FoundationStereo loaded on {DEVICE} from {CKPT_DIR}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_grad_enabled(False)
    load_model()
    yield
    global MODEL
    MODEL = None
    torch.cuda.empty_cache()


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="TidyBot FoundationStereo Service",
    description="Zero-shot stereo depth estimation service for TidyBot. "
    "Accepts rectified stereo pairs and returns disparity + metric depth maps.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────
class DepthRequest(BaseModel):
    left_image: str = Field(..., description="Base64-encoded left rectified image (JPEG/PNG)")
    right_image: str = Field(..., description="Base64-encoded right rectified image (JPEG/PNG)")
    focal_length: float = Field(..., description="Focal length in pixels (fx)")
    baseline: float = Field(..., description="Stereo baseline in meters")
    valid_iters: int = Field(32, description="Number of GRU iterations (higher=more accurate, slower)")
    scale: float = Field(1.0, description="Downscale factor (<=1.0). Use <1 for faster inference on large images.")
    hiera: bool = Field(False, description="Use hierarchical inference (for images >1K resolution)")
    return_disparity: bool = Field(True, description="Return disparity map as base64 float32 numpy array")
    return_depth: bool = Field(True, description="Return metric depth map as base64 float32 numpy array")
    return_vis: bool = Field(False, description="Return disparity visualization as base64 PNG")


class DepthResponse(BaseModel):
    disparity: Optional[str] = Field(None, description="Base64-encoded float32 numpy array (HxW) of disparity in pixels")
    depth: Optional[str] = Field(None, description="Base64-encoded float32 numpy array (HxW) of depth in meters")
    vis: Optional[str] = Field(None, description="Base64-encoded PNG visualization of disparity")
    height: int
    width: int
    inference_ms: float
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    model_loaded: bool
    checkpoint: str
    valid_iters: int


def decode_image(b64: str) -> np.ndarray:
    """Decode base64 image to numpy RGB array."""
    img_data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return np.array(img)


def numpy_to_b64(arr: np.ndarray) -> str:
    """Encode numpy array to base64."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def vis_disparity(disp: np.ndarray) -> np.ndarray:
    """Create a colorized visualization of disparity."""
    disp_vis = disp.copy()
    disp_vis[disp_vis == np.inf] = 0
    disp_vis[disp_vis != disp_vis] = 0  # NaN
    if disp_vis.max() > 0:
        disp_norm = (disp_vis / disp_vis.max() * 255).astype(np.uint8)
    else:
        disp_norm = np.zeros_like(disp_vis, dtype=np.uint8)
    colored = cv2.applyColorMap(disp_norm, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_name = None
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    return HealthResponse(
        status="ok",
        device=DEVICE,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_mem,
        model_loaded=MODEL is not None,
        checkpoint=CKPT_DIR,
        valid_iters=VALID_ITERS,
    )


@app.post("/depth", response_model=DepthResponse)
async def depth(request: DepthRequest):
    """Run stereo depth estimation on a rectified image pair."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        img0 = decode_image(request.left_image)
        img1 = decode_image(request.right_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if img0.shape != img1.shape:
        raise HTTPException(status_code=400, detail=f"Image shapes must match: {img0.shape} vs {img1.shape}")

    scale = request.scale
    if scale < 1.0:
        img0 = cv2.resize(img0, None, fx=scale, fy=scale)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)

    H, W = img0.shape[:2]

    # Convert to tensor
    t0 = torch.as_tensor(img0).to(DEVICE).float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img1).to(DEVICE).float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)

    t_start = time.perf_counter()
    with torch.cuda.amp.autocast(True):
        if not request.hiera:
            disp = MODEL.forward(t0, t1, iters=request.valid_iters, test_mode=True)
        else:
            disp = MODEL.run_hierachical(t0, t1, iters=request.valid_iters, test_mode=True, small_ratio=0.5)
    inference_ms = (time.perf_counter() - t_start) * 1000

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).astype(np.float32)

    # Build response
    resp = DepthResponse(
        height=H,
        width=W,
        inference_ms=round(inference_ms, 2),
        device=DEVICE,
    )

    if request.return_disparity:
        resp.disparity = numpy_to_b64(disp)

    if request.return_depth:
        focal = request.focal_length * scale
        depth_map = (focal * request.baseline) / np.clip(disp, 1e-6, None)
        # Clip unreasonable depths
        depth_map = depth_map.astype(np.float32)
        resp.depth = numpy_to_b64(depth_map)

    if request.return_vis:
        vis_img = vis_disparity(disp)
        _, png_data = cv2.imencode(".png", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        resp.vis = base64.b64encode(png_data.tobytes()).decode()

    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
