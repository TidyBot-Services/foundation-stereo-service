"""
TidyBot FoundationStereo Service — Python Client SDK

Usage:
    from services.foundation_stereo.client import FoundationStereoClient

    client = FoundationStereoClient()
    result = client.depth(left_bytes, right_bytes, focal_length=382.5, baseline=0.055)
    print(f"Depth shape: {result['depth'].shape}, inference: {result['inference_ms']:.1f}ms")
"""

import base64
import io
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import numpy as np


class FoundationStereoClient:
    """Client SDK for the TidyBot FoundationStereo Depth Estimation Service."""

    def __init__(self, host: str = "http://158.130.109.188:8003", timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.host}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.host}{path}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def health(self) -> dict:
        """Check service health and GPU status."""
        return self._get("/health")

    @staticmethod
    def _encode_image(image) -> str:
        """Encode image to base64 from file path, bytes, numpy array, or pass through if already base64."""
        if isinstance(image, np.ndarray):
            import cv2
            _, buf = cv2.imencode(".png", image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf.tobytes()).decode()
        elif isinstance(image, (str, Path)):
            p = Path(image)
            if p.exists():
                return base64.b64encode(p.read_bytes()).decode()
            return image
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        return image

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
            left_image: Left rectified image — file path, raw bytes, numpy array (HxWx3 RGB), or base64.
            right_image: Right rectified image — same formats.
            focal_length: Focal length in pixels.
            baseline: Stereo baseline in meters.
            valid_iters: Number of GRU iterations.
            scale: Downscale factor (<=1.0).
            hiera: Use hierarchical inference for high-res images.
            return_disparity: Include disparity map in response.
            return_depth: Include metric depth map in response.
            return_vis: Include colorized disparity visualization (PNG).

        Returns:
            Dict with depth (np.ndarray HxW float32 meters), disparity (np.ndarray), inference_ms.
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
        data = self._post("/depth", payload)

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

        Returns:
            Same as depth().
        """
        return self.depth(left_image, right_image, focal_length=fx, baseline=baseline, **kwargs)


if __name__ == "__main__":
    client = FoundationStereoClient()
    print("Health:", client.health())
