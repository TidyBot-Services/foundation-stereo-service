"""
TidyBot FoundationStereo Service — Python Client SDK

Usage:
    from client import FoundationStereoClient

    client = FoundationStereoClient("http://<backend-host>:8001")

    # Check service health
    health = client.health()
    print(health)

    # Run stereo depth estimation
    result = client.depth(
        left_image="left.png",
        right_image="right.png",
        focal_length=382.5,  # pixels
        baseline=0.055,      # meters
    )
    print(f"Depth shape: {result['depth'].shape}")
    print(f"Inference: {result['inference_ms']:.1f} ms")
"""

import base64
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import requests


class FoundationStereoClient:
    """Client SDK for the TidyBot FoundationStereo Depth Estimation Service."""

    def __init__(self, base_url: str = "http://localhost:8003", timeout: float = 120.0):
        """
        Args:
            base_url: URL where the FoundationStereo service is hosted.
            timeout: Request timeout in seconds (stereo matching can be slow).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """
        Check service health and GPU status.

        Returns:
            dict with keys: status, device, gpu_name, gpu_memory_mb, model_loaded, checkpoint, valid_iters
        """
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _encode_image(self, image) -> str:
        """Encode image to base64 from file path, bytes, numpy array, or pass through if already base64."""
        if isinstance(image, np.ndarray):
            import cv2
            _, buf = cv2.imencode(".png", image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf.tobytes()).decode()
        elif isinstance(image, (str, Path)):
            return base64.b64encode(Path(image).read_bytes()).decode()
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        return image  # assume already base64

    @staticmethod
    def _decode_numpy(b64: str) -> np.ndarray:
        """Decode base64-encoded numpy array."""
        buf = io.BytesIO(base64.b64decode(b64))
        return np.load(buf)

    def depth(
        self,
        left_image,
        right_image,
        focal_length: float,
        baseline: float,
        valid_iters: int = 32,
        scale: float = 1.0,
        hiera: bool = False,
        return_disparity: bool = True,
        return_depth: bool = True,
        return_vis: bool = False,
    ) -> dict:
        """
        Run stereo depth estimation on a rectified image pair.

        Args:
            left_image: Left rectified image — file path (str/Path), raw bytes, numpy array (HxWx3 RGB), or base64.
            right_image: Right rectified image — same formats as left_image.
            focal_length: Focal length in pixels (fx from camera intrinsics).
            baseline: Stereo baseline in meters.
            valid_iters: Number of GRU iterations (default 32). Higher = better quality, slower.
            scale: Downscale factor (<=1.0). Useful for faster inference on large images.
            hiera: Use hierarchical inference for high-res images (>1K).
            return_disparity: Include disparity map in response.
            return_depth: Include metric depth map in response.
            return_vis: Include colorized disparity visualization (PNG) in response.

        Returns:
            dict with keys:
                - disparity: np.ndarray (HxW, float32) — disparity in pixels (if requested)
                - depth: np.ndarray (HxW, float32) — depth in meters (if requested)
                - vis: bytes — PNG image of colorized disparity (if requested)
                - height: int
                - width: int
                - inference_ms: float
                - device: str
        """
        payload = {
            "left_image": self._encode_image(left_image),
            "right_image": self._encode_image(right_image),
            "focal_length": focal_length,
            "baseline": baseline,
            "valid_iters": valid_iters,
            "scale": scale,
            "hiera": hiera,
            "return_disparity": return_disparity,
            "return_depth": return_depth,
            "return_vis": return_vis,
        }
        r = requests.post(f"{self.base_url}/depth", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        result = {
            "height": data["height"],
            "width": data["width"],
            "inference_ms": data["inference_ms"],
            "device": data["device"],
        }

        if data.get("disparity"):
            result["disparity"] = self._decode_numpy(data["disparity"])
        if data.get("depth"):
            result["depth"] = self._decode_numpy(data["depth"])
        if data.get("vis"):
            result["vis"] = base64.b64decode(data["vis"])

        return result

    def depth_from_realsense(
        self,
        left_image,
        right_image,
        fx: float = 382.545,
        baseline: float = 0.055,
        **kwargs,
    ) -> dict:
        """
        Convenience method for Intel RealSense D435/D455 cameras.

        Args:
            left_image: Left IR or rectified RGB image.
            right_image: Right IR or rectified RGB image.
            fx: Focal length in pixels (default: D435 at 640x480).
            baseline: Stereo baseline in meters (default: D435 ~55mm).
            **kwargs: Additional arguments passed to depth().

        Returns:
            Same as depth().
        """
        return self.depth(left_image, right_image, focal_length=fx, baseline=baseline, **kwargs)


if __name__ == "__main__":
    client = FoundationStereoClient()
    print("Health:", client.health())
