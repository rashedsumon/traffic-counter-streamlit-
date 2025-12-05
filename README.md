# Traffic Counter - Streamlit app

## What this project does
- Detects and tracks vehicles in 4-way intersection videos.
- Maintains four independent counters (North, South, East, West).
- Exports an MP4 with colored bounding boxes and live counts.
- Optional: classify vehicle types (car, truck, bus, motorcycle).
- Optional: exclude pedestrians from vehicle counts.

## Quick start
1. Create Python 3.11 virtualenv and install requirements:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
