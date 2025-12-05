"""
app.py
Streamlit app to run the traffic video processor.
"""

import streamlit as st
from data_loader import download_dataset
from processor import VideoProcessor
import tempfile, os, cv2
from pathlib import Path


st.set_page_config(page_title="Traffic Counter", layout="wide")
st.title("Traffic Counter — 4-way intersection analyzer")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Input")
use_sample = st.sidebar.checkbox("Use sample dataset from kagglehub", value=True)
uploaded_file = None

if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload video file (mp4/mov)", 
                                             type=["mp4", "mov", "avi", "mkv"])

st.sidebar.header("Options")
model_choice = st.sidebar.selectbox("YOLO model", 
                                    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
conf_thresh = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.35)
exclude_ped = st.sidebar.checkbox("Exclude pedestrians", value=True)
max_frames = st.sidebar.number_input("Max frames (0 = full video)", min_value=0, value=0)


# -------------------------
# Dataset download button
# -------------------------
if use_sample:
    st.sidebar.markdown("**Dataset downloader**")
    if st.sidebar.button("Download sample dataset (kagglehub)"):
        with st.spinner("Downloading dataset..."):
            dataset_path = download_dataset("arshadrahmanziban/traffic-video-dataset")
            st.success(f"Downloaded to: {dataset_path}")


# -------------------------
# Select input video
# -------------------------
st.header("Run analysis")
col1, col2 = st.columns([1, 1])

with col1:

    if use_sample:
        st.info("Using sample dataset. Select a video from downloaded folder:")

        # KaggleHub cached path
        kaggle_cached_root = Path.home() / ".cache" / "kagglehub" / "datasets"
        video_files = list(kaggle_cached_root.rglob("*.mp4"))

        if video_files:
            file_options = [""] + [str(p) for p in video_files]
            selected_sample = st.selectbox("Choose a sample video:", file_options)
        else:
            selected_sample = ""
            st.warning("No sample videos found. Download using the button above.")

    else:
        selected_sample = ""

    # Uploaded file option
    if uploaded_file:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(uploaded_file.getbuffer())
        input_path = temp.name
        st.success(f"Uploaded & saved to temporary file.")
    elif use_sample and selected_sample:
        input_path = selected_sample
    else:
        input_path = None

    start = st.button("Start analysis")


# -------------------------
# Output config
# -------------------------
with col2:
    out_dir = st.text_input("Output directory", value="output_videos")
    os.makedirs(out_dir, exist_ok=True)
    st.markdown("Processed video and counts will appear below.")


# -------------------------
# Run processing
# -------------------------
if start:

    if not input_path:
        st.warning("Please select or upload a video.")
        st.stop()

    out_file = os.path.join(out_dir, f"processed_{Path(input_path).stem}.mp4")

    processor = VideoProcessor(model_path=model_choice,
                               device=device,
                               conf_thres=conf_thresh,
                               exclude_pedestrians=exclude_ped)

    st.info("Processing video — this may take time depending on the size.")
    with st.spinner("Analyzing..."):
        max_f = int(max_frames) if max_frames > 0 else None
        output_path, counts = processor.process(input_path, out_file, 
                                                max_frames=max_f,
                                                show_progress=True)

    st.success("Processing complete!")

    # ---------------------
    # Video Preview (OpenCV snapshot)
    # ---------------------
    st.markdown("### Video Preview (first frame)")
    cap = cv2.VideoCapture(output_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Preview Frame")
    else:
        st.warning("Preview not available.")
    cap.release()

    # ---------------------
    # Final counts
    # ---------------------
    st.markdown("### Final Counts")
    st.write(counts)

    # ---------------------
    # Download button
    # ---------------------
    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", 
                           data=f, 
                           file_name=os.path.basename(output_path),
                           mime="video/mp4")
