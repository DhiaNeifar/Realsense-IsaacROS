import importlib.util

import numpy as np
import pytest


HAS_CV2 = importlib.util.find_spec("cv2") is not None


@pytest.mark.skipif(not HAS_CV2, reason="cv2 is not installed in this environment")
def test_depth_to_meters_converts_uint16_millimeters():
    from realsense_benchmark.depth_tools import depth_to_meters

    depth_raw = np.array([[500, 1250]], dtype=np.uint16)

    depth_m = depth_to_meters(depth_raw)

    assert depth_m.dtype == np.float32
    assert np.allclose(depth_m, np.array([[0.5, 1.25]], dtype=np.float32))


@pytest.mark.skipif(not HAS_CV2, reason="cv2 is not installed in this environment")
def test_render_depth_band_overlay_returns_colorized_image():
    from realsense_benchmark.depth_tools import render_depth_band_overlay

    depth_raw = np.array(
        [
            [300, 500, 900],
            [450, 800, 1400],
        ],
        dtype=np.uint16,
    )

    overlay, proc_ms = render_depth_band_overlay(
        depth_raw,
        band_min_m=0.4,
        band_max_m=1.0,
        cpu_loops=1,
    )

    assert overlay.shape == (2, 3, 3)
    assert overlay.dtype == np.uint8
    assert proc_ms >= 0.0
