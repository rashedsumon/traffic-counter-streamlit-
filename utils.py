"""
utils.py
Drawing helpers, counting logic and configuration defaults.
"""

import cv2
from collections import defaultdict

# Colors for drawing (BGR)
COLOR_MAP = {
    "car": (0, 255, 0),
    "truck": (0, 165, 255),
    "bus": (0, 0, 255),
    "motorcycle": (255, 0, 0),
    "person": (128, 128, 128),
    "default": (255, 255, 255)
}

VEHICLE_CLASS_NAME_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def draw_overlay(frame, tracks, counters, fps=None, show_ids=True):
    """
    Draw bounding boxes, ids and live counters on the frame.
    tracks: list of {'id','bbox','centroid','class_name'}
    counters: dict with keys ['North','South','East','West']
    """
    h, w = frame.shape[:2]
    # draw center lines (visual aid)
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx, 0), (cx, h), (200,200,200), 1)
    cv2.line(frame, (0, cy), (w, cy), (200,200,200), 1)

    # draw each track
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        class_name = VEHICLE_CLASS_NAME_MAP.get(t["class_id"], "default") if t["class_id"] is not None else "default"
        color = COLOR_MAP.get(class_name, COLOR_MAP["default"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f'ID:{t["id"]}'
        if t.get("class_name"):
            text += f' {t.get("class_name")}'
        if show_ids:
            cv2.putText(frame, text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # draw counters at top-left
    y0 = 20
    for i, direction in enumerate(["North", "South", "East", "West"]):
        cv2.putText(frame, f'{direction}: {counters.get(direction,0)}', (10, y0 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # optional fps
    if fps is not None:
        cv2.putText(frame, f'FPS: {fps:.1f}', (frame.shape[1]-120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    return frame

class DirectionCounter:
    """
    Maintains counts per direction using a simple rule:
    - When a new track id is first seen, record its first centroid.
    - When the track disappears (or when we optionally confirm crossing), determine the entry direction by
      comparing the first centroid with the frame center:
        y < center_y --> North
        y > center_y --> South
        x < center_x --> West
        x > center_x --> East
    This is simple and works well for cameras capturing the full intersection from one side. For angled cameras,
    replace with line-crossing or ROI membership logic.
    """

    def __init__(self, frame_width, frame_height, exclude_pedestrians=True):
        self.first_position = {}   # id -> (cx,cy)
        self.seen_ids = set()
        self.counters = {"North":0, "South":0, "East":0, "West":0}
        self.w = frame_width
        self.h = frame_height
        self.exclude_pedestrians = exclude_pedestrians
        # keep last-known position to detect crossing if desired
        self.last_position = {}

    def update_tracks(self, tracks):
        """
        tracks: list of tracks (each contains 'id','centroid','class_id')
        This registers new ids and updates last positions.
        """
        current_ids = set()
        for t in tracks:
            tid = t["id"]
            current_ids.add(tid)
            cx, cy = t["centroid"]
            if tid not in self.first_position:
                # optional: ignore pedestrian initial detection
                if self.exclude_pedestrians and t.get("class_id") == 0:
                    # mark ignored
                    self.first_position[tid] = ("ignored", (cx,cy))
                else:
                    self.first_position[tid] = ((cx,cy))
            self.last_position[tid] = (cx,cy)
        # detect disappeared tracks (ids seen before but not this frame)
        disappeared = set(self.first_position.keys()).difference(current_ids)
        # NB: we do not remove 'ignored' entries unless confirmed removed
        for tid in list(disappeared):
            # if ignored, remove without counting
            fp = self.first_position.get(tid)
            if isinstance(fp, tuple) and fp and fp[0] == "ignored":
                # cleanup
                del self.first_position[tid]
                if tid in self.last_position:
                    del self.last_position[tid]
                continue
            # get first position
            first_xy = self.first_position.get(tid)
            if first_xy and isinstance(first_xy, tuple):
                fx, fy = first_xy
                # decide direction
                cx, cy = self.w//2, self.h//2
                # determine primary axis
                dx = fx - cx
                dy = fy - cy
                # choose largest absolute component to determine which side
                if abs(dx) > abs(dy):
                    # left or right
                    if dx < 0:
                        self.counters["West"] += 1
                    else:
                        self.counters["East"] += 1
                else:
                    if dy < 0:
                        self.counters["North"] += 1
                    else:
                        self.counters["South"] += 1
            # cleanup
            if tid in self.first_position:
                del self.first_position[tid]
            if tid in self.last_position:
                del self.last_position[tid]

    def get_counts(self):
        return dict(self.counters)

    def reset(self):
        self.first_position.clear()
        self.last_position.clear()
        self.counters = {"North":0, "South":0, "East":0, "West":0}
