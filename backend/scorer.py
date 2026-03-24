"""Convert normalized dartboard coordinates to scores.

The warped+cropped image has the board centered at (0.5, 0.5) and
filling the frame. Coordinates are in [0, 1] normalized space.

PDC standard dartboard: 451mm outer diameter.
Ring radii (mm from center): Bull 6.35, Outer Bull 15.9,
Inner Single 99, Triple 107, Outer Single 162, Double 170.

Sectors clockwise from top: 20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
3, 19, 7, 16, 8, 11, 14, 9, 12, 5. The 20 segment is at the top,
offset 9 degrees clockwise from the vertical.
"""

import math

BOARD_RADIUS_MM = 225.5
BULL_RADIUS = 6.35 / BOARD_RADIUS_MM
OUTER_BULL_RADIUS = 15.9 / BOARD_RADIUS_MM
TRIPLE_INNER_RADIUS = 99.0 / BOARD_RADIUS_MM
TRIPLE_OUTER_RADIUS = 107.0 / BOARD_RADIUS_MM
DOUBLE_INNER_RADIUS = 162.0 / BOARD_RADIUS_MM
DOUBLE_OUTER_RADIUS = 170.0 / BOARD_RADIUS_MM

SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
SECTOR_ANGLE = math.radians(18)  # 360 / 20
ROTATION_OFFSET = math.radians(9)  # 20 is 9 degrees clockwise from top


def score_dart(x_norm: float, y_norm: float) -> dict:
    """Convert normalized (x, y) to a dartboard score.

    Args:
        x_norm: horizontal position in [0, 1], 0.5 = center
        y_norm: vertical position in [0, 1], 0.5 = center

    Returns:
        dict with keys: segment, multiplier, score, label
    """
    dx = (x_norm - 0.5) * 2  # [-1, 1]
    dy = (y_norm - 0.5) * 2  # [-1, 1]
    r = math.sqrt(dx * dx + dy * dy)  # 0 = center, 1 = board edge

    # Bull / Outer Bull
    if r <= BULL_RADIUS:
        return {"segment": 25, "multiplier": 2, "score": 50, "label": "Bull"}
    if r <= OUTER_BULL_RADIUS:
        return {"segment": 25, "multiplier": 1, "score": 25, "label": "Outer Bull"}

    # Miss (outside the board)
    if r > DOUBLE_OUTER_RADIUS:
        return {"segment": 0, "multiplier": 0, "score": 0, "label": "Miss"}

    # Determine sector from angle
    # atan2 gives angle from positive x-axis, counter-clockwise
    # We want angle from top (negative y-axis), clockwise
    angle = math.atan2(dx, -dy)  # 0 = top, positive = clockwise
    angle = (angle + ROTATION_OFFSET) % (2 * math.pi)
    sector_idx = int(angle / SECTOR_ANGLE) % 20
    segment = SECTOR_ORDER[sector_idx]

    # Determine ring (multiplier)
    if r <= TRIPLE_INNER_RADIUS:
        multiplier = 1
    elif r <= TRIPLE_OUTER_RADIUS:
        multiplier = 3
    elif r <= DOUBLE_INNER_RADIUS:
        multiplier = 1
    else:
        multiplier = 2

    score = segment * multiplier
    if multiplier == 3:
        label = f"T{segment}"
    elif multiplier == 2:
        label = f"D{segment}"
    else:
        label = str(segment)

    return {"segment": segment, "multiplier": multiplier, "score": score, "label": label}
