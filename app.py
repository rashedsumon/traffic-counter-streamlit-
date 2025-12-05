"""
app.py
Streamlit app to run the traffic video processor.
"""

import streamlit as st
from data_loader import download_dataset
from processor import VideoProcessor
import tempfile
import os
import cv2
from pathlib import Path

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Traffic Counter", layout="wide")
st.title("Traffic Counter — 4-way Intersection Analyzer")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Input Source")
use_sample = st.sidebar.checkbox("Use sample dataset from KaggleHub", value=True)
uploaded_file = None

if not use_sample:
    uploaded_file = st.sidebar.file_uploader(
        "Upload video file (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"]
    )

st.sidebar.header("Options")
model_choice = st.sidebar.selectbox(
    "YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0
)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
conf_thresh = st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.35)
exclude_ped = st.sidebar.checkbox("Exclude Pedestrians", value=True)
max_frames = st.sidebar.number_input(
    "Max Frames (0 = full video)", min_value=0, value=0
)

# -------------------------
# Sample Dataset Download
# -------------------------
if use_sample:
    st.sidebar.markdown("**Dataset Downloader**")
    if st.sidebar.button("Download Sample Dataset (KaggleHub)"):
        with st.spinner("Downloading dataset..."):
            dataset_path = download_dataset("arshadrahmanziban/traffic-video-dataset")
            st.success(f"Downloaded to: {dataset_path}")

# -------------------------
# Select Input Video
# -------------------------
st.header("Run Analysis")
col1, col2 = st.columns([1, 1])

with col1:
    input_path = None

    if use_sample:
        st.info("Using sample dataset. Select a video from downloaded folder:")

        kaggle_cached_root = Path.home() / ".cache" / "kagglehub" / "datasets"
        video_files = list(kaggle_cached_root.rglob("*.mp4"))

        if video_files:
            file_options = [""] + [str(p) for p in video_files]
            selected_sample = st.selectbox("Choose a sample video:", file_options)
            if selected_sample:
                input_path = selected_sample
        else:
            st.warning("No sample videos found. Download using the button above.")

    # Uploaded file
    if uploaded_file:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(uploaded_file.getbuffer())
        input_path = temp.name
        st.success(f"Uploaded & saved to temporary file: {input_path}")

    start = st.button("Start Analysis")

# -------------------------
# Output Config
# -------------------------
with col2:
    out_dir = st.text_input("Output Directory", value="output_videos")
    os.makedirs(out_dir, exist_ok=True)
    st.markdown("Processed video and counts will appear below.")

# -------------------------
# Run Processing
# -------------------------
if start:
    if not input_path:
        st.warning("Please select or upload a video.")
        st.stop()

    out_file = os.path.join(out_dir, f"processed_{Path(input_path).stem}.mp4")

    processor = VideoProcessor(
        model_path=model_choice,
        device=device,
        conf_thres=conf_thresh,
        exclude_pedestrians=exclude_ped
    )

    st.info("Processing video — this may take time depending on size.")
    with st.spinner("Analyzing..."):
        max_f = int(max_frames) if max_frames > 0 else None
        output_path, counts = processor.process(
            input_path, out_file, max_frames=max_f, show_progress=True
        )

    st.success("Processing complete!")

    # ---------------------
    # Video Preview (first frame)
    # ---------------------
    st.markdown("### Video Preview (First Frame)")
    cap = cv2.VideoCapture(output_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Preview Frame")
    else:
        st.warning("Preview not available.")
    cap.release()

    # ---------------------
    # Final Counts
    # ---------------------
    st.markdown("### Vehicle Counts")
    st.write(counts)

    # ---------------------
    # Download Button
    # ---------------------
    with open(output_path, "rb") as f:
        st.download_button(
            "Download Processed Video",
            data=f,
            file_name=os.path.basename(output_path),
            mime="video/mp4"
        )
