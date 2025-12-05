"""
detector.py
YOLOv8 detection wrapper using ultralytics.

Provides:
 - load_model(model_name_or_path)
 - detect(frame) -> list of detections (x1,y1,x2,y2,confidence,class_id,class_name)
"""

from ultralytics import YOLO
import numpy as np

# default model (will auto-download the small model the first run)
DEFAULT_MODEL = "yolov8n.pt"

# mapping of COCO class ids (ultralytics default) to names
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    # ... (we only need certain classes)
}

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
PEDESTRIAN_CLASS_ID = 0

class YOLODetector:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cpu", conf_threshold: float = 0.35):
        """
        model_path: model name (yolov8n.pt) or path to weights
        device: "cpu" or "cuda"
        conf_threshold: detections below this are discarded
        """
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Run model on one frame (numpy BGR).
        Returns list of dicts: {'bbox':(x1,y1,x2,y2),'conf':f,'class_id':int,'class_name':str}
        """
        # ultralytics expects RGB
        results = self.model(frame[:, :, ::-1], imgsz=1280, verbose=False)
        detections = []
        # results is list with one element
        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return []
        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            if conf < self.conf_threshold:
                continue
            # Optional: filter to vehicles or pedestrians, but we return all for flexibility
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            name = COCO_NAMES.get(cls, str(cls))
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "class_id": cls,
                "class_name": name
            })
        return detections

if __name__ == "__main__":
    # quick local test (requires a frame)
    pass
