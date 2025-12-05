"""
tracker.py
Simple wrapper around norfair for tracking bounding boxes across frames.
"""

from norfair import Detection, Tracker
import numpy as np

# Norfair distance function: Euclidean on centroids
def _p_dist(detection, tracked_object):
    # detection.points and tracked_object.points are arrays
    return np.linalg.norm(detection.points - tracked_object.estimate)

class NorfairTrackerWrapper:
    def __init__(self, distance_threshold=30, hit_inertia_min=3, initialization_delay=0):
        """
        distance_threshold: pixel threshold for matching detections to tracks
        hit_inertia_min: how many frames a track must be seen before considered "stable"
        """
        self.tracker = Tracker(distance_function=_p_dist,
                               distance_threshold=distance_threshold,
                               hit_inertia_min=hit_inertia_min,
                               initialization_delay=initialization_delay)

    def update(self, detections):
        """
        detections: list of dicts with 'bbox':(x1,y1,x2,y2)
        returns list of tracked objects: [{'id':int,'bbox':(x1,y1,x2,y2),'last_position':(cx,cy)}]
        """
        norfair_dets = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            norfair_dets.append(Detection(points=np.array([[cx, cy]], dtype=np.float32),
                                          scores=np.array([d["conf"]], dtype=np.float32),
                                          data=d))  # keep original detection in data

        tracked = self.tracker.update(detections=norfair_dets)
        out = []
        for t in tracked:
            # t.estimate is centroid
            est = t.estimate[0]
            # Norfair doesn't store bbox for tracked object by default; we use associated detection data if exists
            # t.last_detection may be available; otherwise store centroid only
            bbox = None
            if hasattr(t, "last_detection") and t.last_detection is not None:
                data = t.last_detection.data
                bbox = data["bbox"]
                class_id = data["class_id"]
                class_name = data["class_name"]
                conf = data["conf"]
            else:
                bbox = (int(est[0])-10, int(est[1])-10, int(est[0])+10, int(est[1])+10)
                class_id = None
                class_name = None
                conf = None
            out.append({
                "id": int(t.id),
                "bbox": tuple(map(int, bbox)),
                "centroid": (int(est[0]), int(est[1])),
                "class_id": class_id,
                "class_name": class_name,
                "conf": conf
            })
        return out
