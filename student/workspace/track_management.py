"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    transform = np.asarray(meas.sensor.sens_to_veh)
    rot = transform[:3, :3]
    trans = transform[:3, 3]

    pos_sens = np.asarray(meas.z, dtype=float).reshape(3)
    pos_veh = rot @ pos_sens + trans

    x = np.zeros((6, 1), dtype=float)
    x[:3, 0] = pos_veh

    P = np.zeros((6, 6), dtype=float)
    P[:3, :3] = rot @ np.asarray(meas.R) @ rot.T
    P[3, 3] = params.sigma_p44**2
    P[4, 4] = params.sigma_p55**2
    P[5, 5] = params.sigma_p66**2

    score = 1.0 / params.window
    state = "initialized"

    return {
        "x": np.asmatrix(x),
        "P": np.asmatrix(P),
        "score": float(score),
        "state": state,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    step = 1.0 / params.window
    current_score = float(track["score"])
    current_state = track.get("state", "tentative")

    if associated:
        new_score = min(1.0, current_score + step)
    else:
        new_score = current_score - step

    if new_score > params.confirmed_threshold:
        new_state = "confirmed"
    elif current_state == "confirmed":
        new_state = "confirmed"
    else:
        new_state = "tentative"

    track["score"] = new_score
    track["state"] = new_state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    P = np.asarray(track["P"])
    score = float(track["score"])
    state = track.get("state", "tentative")

    # 1. Pxx or Pyy > max_P: delete regardless of score
    if P[0, 0] > params.max_P or P[1, 1] > params.max_P:
        return True

    # 2. Confirmed: delete when score < delete_threshold
    if state == "confirmed":
        if score < params.delete_threshold:
            return True
    else:
        # 3. Unconfirmed: delete when score <= 0
        if score <= 0.0:
            return True

    return False
