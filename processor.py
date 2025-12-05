"""
processor.py
Processing pipeline: open video, run detector -> tracker -> counter -> draw -> write output
"""

import cv2
import time
from detector import YOLODetector
from tracker import NorfairTrackerWrapper
from utils import draw_overlay, DirectionCounter
import os
from moviepy.editor import VideoFileClip

class VideoProcessor:
    def __init__(self, model_path=None, device="cpu", conf_thres=0.35, exclude_pedestrians=True):
        self.detector = YOLODetector(model_path or "yolov8n.pt", device=device, conf_threshold=conf_thres)
        self.tracker = NorfairTrackerWrapper(distance_threshold=40, hit_inertia_min=3)
        self.exclude_pedestrians = exclude_pedestrians

    def process(self, input_path, output_path, max_frames=None, show_progress=False):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

        counter = DirectionCounter(frame_width=w, frame_height=h, exclude_pedestrians=self.exclude_pedestrians)

        frame_idx = 0
        t0 = time.time()
        last_time = t0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            # optionally limit frames for testing
            if max_frames and frame_idx > max_frames:
                break

            # detection
            detections = self.detector.detect(frame)
            # filter out pedestrians if exclude_pedestrians is True
            if self.exclude_pedestrians:
                detections = [d for d in detections if d["class_id"] != 0]

            # update tracker
            tracks = self.tracker.update(detections)

            # update counters with tracks
            counter.update_tracks(tracks)

            # draw overlay
            elapsed = time.time() - last_time
            fps_local = 1.0 / elapsed if elapsed > 0 else fps
            out_frame = draw_overlay(frame.copy(), tracks, counter.get_counts(), fps=fps_local)
            last_time = time.time()

            writer.write(out_frame)

            if show_progress and frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

        cap.release()
        writer.release()
        total_elapsed = time.time() - t0
        print(f"Finished processing. Output saved to {output_path}. Time: {total_elapsed:.1f}s, frames: {frame_idx}")
        return output_path, counter.get_counts()
