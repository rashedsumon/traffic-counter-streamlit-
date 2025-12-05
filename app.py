"""
app.py
Streamlit app to run the traffic video processor.

Main features:
- Upload local video or use dataset downloaded by data_loader
- Set options (classification, pedestrian exclusion)
- Start analysis and show resulting video + counts + download link
"""

import streamlit as st
from data_loader import download_dataset
from processor import VideoProcessor
import tempfile, os
from pathlib import Path
from moviepy.editor import VideoFileClip

st.set_page_config(page_title="Traffic Counter", layout="wide")

st.title("Traffic Counter — 4-way intersection analyzer")

st.sidebar.header("Input")
use_sample = st.sidebar.checkbox("Use sample dataset from kagglehub", value=True)
uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload video file (mp4/mov)", type=["mp4","mov","avi","mkv"])

st.sidebar.header("Options")
model_choice = st.sidebar.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
conf_thresh = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.35)
exclude_ped = st.sidebar.checkbox("Exclude pedestrians from vehicle counts", value=True)
max_frames = st.sidebar.number_input("Max frames for quick test (0 = full video)", min_value=0, value=0)

if use_sample:
    st.sidebar.markdown("**Dataset downloader**")
    if st.sidebar.button("Download sample dataset (kagglehub)"):
        with st.spinner("Downloading dataset from kagglehub..."):
            path = download_dataset("arshadrahmanziban/traffic-video-dataset")
            st.success(f"Downloaded to {path}")

# Main area
st.header("Run analysis")
col1, col2 = st.columns([1,1])

with col1:
    if use_sample:
        st.info("Using dataset sample. Select a file from datasets/ after download, or upload a different video.")
        sample_dir = Path("datasets")
        files = []
        if sample_dir.exists():
            files = [str(p) for p in sample_dir.rglob("*.mp4")]
        selected_sample = st.selectbox("Choose a sample video (if available)", options=[""] + files)
    else:
        selected_sample = ""

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.getbuffer())
        input_path = tfile.name
        st.success(f"Saved uploaded file to {input_path}")
    elif use_sample and selected_sample:
        input_path = selected_sample
    else:
        input_path = None

    st.markdown("**Start processing**")
    start = st.button("Start analysis")

with col2:
    st.markdown("**Output & Controls**")
    out_dir = st.text_input("Output directory", value="output_videos")
    os.makedirs(out_dir, exist_ok=True)
    st.markdown("When finished, you will get a downloadable MP4 and the final counts.")

if start:
    if not input_path:
        st.warning("No input video selected. Upload or select a sample.")
    else:
        out_file = os.path.join(out_dir, f"processed_{Path(input_path).stem}.mp4")
        processor = VideoProcessor(model_path=model_choice, device=device, conf_thres=conf_thresh, exclude_pedestrians=exclude_ped)
        st.info("Processing started. This runs in this session — large videos may take time and CPU.")
        with st.spinner("Analyzing video... this may take a while"):
            max_f = int(max_frames) if max_frames > 0 else None
            output_path, counts = processor.process(input_path, out_file, max_frames=max_f, show_progress=True)
        st.success(f"Processing finished: {output_path}")

        # show video preview using moviepy to ensure playable file
        try:
            clip = VideoFileClip(output_path)
            preview_path = os.path.join(out_dir, f"preview_{Path(output_path).stem}.mp4")
            # ensure small preview size (first 20 seconds)
            duration = min(clip.duration, 20)
            clip.subclip(0, duration).write_videofile(preview_path, audio=False, verbose=False, logger=None)
            st.video(preview_path)
        except Exception as e:
            st.error(f"Could not create preview: {e}")

        st.markdown("### Final counts")
        st.write(counts)
        with open(output_path, "rb") as f:
            st.download_button("Download processed video", data=f, file_name=os.path.basename(output_path), mime="video/mp4")
