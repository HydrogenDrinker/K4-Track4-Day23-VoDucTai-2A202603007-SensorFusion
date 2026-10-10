"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    pos_veh = np.asarray(x, dtype=float).reshape(-1)[:3]
    if not np.isfinite(pos_veh).all():
        return False
    transform = np.asarray(sensor.veh_to_sens)
    pos_sens = transform[:3, :3] @ pos_veh + transform[:3, 3]
    if not np.isfinite(pos_sens).all():
        return False

    depth_eps = getattr(sensor, "depth_epsilon", 1e-6)
    if sensor.name == "camera":
        if pos_sens[0] <= depth_eps:
            return False

    angle = float(np.arctan2(pos_sens[1], pos_sens[0]))
    min_fov, max_fov = min(sensor.fov), max(sensor.fov)
    return bool(min_fov <= angle <= max_fov)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    pos_veh = np.asarray(x, dtype=float).reshape(-1)[:3]
    depth_eps = getattr(sensor, "depth_epsilon", 1e-6)
    if not np.isfinite(pos_veh).all():
        raise ValueError(f"Camera coordinates nonfinite: vehicle pos={pos_veh}")
    transform = np.asarray(sensor.veh_to_sens)
    pos_sens = transform[:3, :3] @ pos_veh + transform[:3, 3]
    if not np.isfinite(pos_sens).all() or pos_sens[0] <= depth_eps:
        raise ValueError(
            f"Camera projection needs finite coordinates and positive depth > {depth_eps:g}; "
            f"sensor position={pos_sens.tolist()}"
        )
    x_s, y_s, z_s = pos_sens
    u = sensor.c_i - sensor.f_i * (y_s / x_s)
    v = sensor.c_j - sensor.f_j * (z_s / x_s)
    return np.asmatrix([[u], [v]], dtype=float)


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    z_arr = np.asarray(z, dtype=float).reshape(2, 1)
    z_mat = np.asmatrix(z_arr)
    R = np.asmatrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2]))
    return {"z": z_mat, "R": R, "sensor": sensor}
