# Bottle Classification System

A desktop application for automated bottle inspection using PyQt5, OpenCV, and YOLO models. The system analyzes images, videos, and live webcam streams to detect bottles and classify each one by deformity, fullness, and capacity, while also tracking unique bottles for counting in video-based workflows.

## Overview

This project combines a professional desktop interface with a multi-model computer vision pipeline:

- `app.py` provides the PyQt5 desktop application and user interface.
- `pipeline.py` runs the inference pipeline using three YOLO models.
- `counter.py` handles bottle grouping, frame-to-frame tracking, and counting logic.
- `smoke_test.py` provides a minimal pipeline sanity check.

The application is designed for interactive inspection and review. Users can upload a single image, process a full video, or open a webcam feed for real-time analysis.

## Features

- Desktop GUI built with PyQt5
- Support for image, video, and webcam input
- Three-model YOLO inference pipeline
- Bottle classification by:
  - deformity
  - fullness
  - capacity
- Unique bottle counting for video and webcam streams
- Annotated output preview with overlays
- Run history with saved session summaries
- Per-run statistics and aggregated statistics across all runs
- Export of annotated images and videos

## How It Works

For each frame, the system:

1. Runs three YOLO models in parallel.
2. Collects detections from the deformity, fullness, and capacity models.
3. Groups nearby detections so one physical bottle is represented once.
4. Re-checks grouped bottle crops to refine labels.
5. Draws annotations on the output frame.
6. Tracks bottles across frames for video and webcam modes.
7. Counts a bottle only when it crosses the vertical counting line.

This design helps reduce duplicate counts and stabilizes bottle attributes across frames.

## Interface Summary

The application includes these main tabs:

- `Dashboard`: media upload, live preview, and model output text
- `Results`: annotated output, run metadata, save actions, and detection table
- `Run History`: previously processed runs with quick open/save actions
- `Total Stats`: charts aggregated across all runs
- `Stats`: charts for the selected run
- `Instructions`: in-app usage guidance

## Project Structure

```text
Bottle Classification System/
|-- app.py
|-- pipeline.py
|-- counter.py
|-- smoke_test.py
|-- requirements.txt
|-- capacity.pt
|-- deformity.pt
|-- fullness.pt
|-- annotated_video.mp4
|-- .mpl_config/
|-- .yolo_config/
`-- __pycache__/
```

## Requirements

- Python 3.11 recommended
- Windows environment is the primary tested target in this repository
- A webcam is required for live mode

Python dependencies are listed in `requirements.txt`:

- `PyQt5`
- `opencv-python`
- `numpy`
- `ultralytics`
- `torch`
- `torchvision`
- `matplotlib`
- `lap`

## Installation

1. Open a terminal in the project folder.
2. Create and activate a virtual environment.
3. Install the project dependencies.

```powershell
cd "C:\path\to\Bottle Classification System"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Step-by-Step Setup and Run

If you downloaded the project as a ZIP or copied the folder manually, use this full setup flow:

1. Open `Bottle Classification System` in your terminal or IDE.
2. Make sure Python 3.11 is installed.
3. Create a virtual environment:

```powershell
python -m venv .venv
```

4. Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

5. Install all required libraries:

```powershell
pip install -r requirements.txt
```

6. Confirm the model files exist in the folder:

- `deformity.pt`
- `fullness.pt`
- `capacity.pt`

7. Run the application:

```powershell
python app.py
```

8. In the app, choose one of the following:

- `Upload Image` to analyze a single bottle image
- `Upload Video` to process a full video and generate an annotated output video
- `Open Webcam` to run live bottle detection and counting

9. Review the results in the `Results`, `Run History`, `Stats`, and `Total Stats` tabs.
10. Save annotated images or videos from the `Results` tab when needed.

## Model Files

The repository expects these model files to be present:

- `deformity.pt`
- `fullness.pt`
- `capacity.pt`

`pipeline.py` can also resolve equivalent model files from a `models/` directory if you later reorganize the project.

## Running the Application

Start the desktop application with:

```powershell
python app.py
```

If the required libraries are installed and the model files are present, the main desktop window should open and the system will be ready for image, video, or webcam processing.

## Usage

### Image Mode

- Click `Upload Image`
- Select an image file
- Review the annotated output and detected bottle attributes
- Save the annotated result from the `Results` tab

### Video Mode

- Click `Upload Video`
- Select a video file
- Wait for full processing to complete
- Review the counted unique bottles and generated annotated video
- Save the processed video from the `Results` tab

### Webcam Mode

- Click `Open Webcam`
- Present bottles to the camera
- Watch the live annotations and unique bottle counter
- Click `Stop` to finalize the run and save its results in history

## Output and Results

The application stores each run in memory and exposes:

- start and end timestamps
- source type
- total unique bottles
- processing time
- per-bottle classification table
- per-run charts
- aggregated charts across all runs

For video processing, the application writes an annotated output video to:

```text
annotated_video.mp4
```

## Smoke Test

A basic smoke test is included:

```powershell
python smoke_test.py
```

## Troubleshooting

### Model loading error with Ultralytics

During verification in this environment, the smoke test failed while loading the YOLO weights with:

```text
AttributeError: Can't get attribute 'C3k2' on ultralytics.nn.modules.block
```

This usually means the installed `ultralytics` version does not match the version used when the `.pt` weights were trained or exported.

Recommended fixes:

- install the same `ultralytics` version used to train/export the weights
- re-export the model weights with your current toolchain
- pin dependency versions once the correct combination is confirmed

## Notes for GitHub

- The repository contains large binary assets, including three model files of about 44 MB each.
- The sample `annotated_video.mp4` file is also included and is about 22 MB.
- These files are below GitHub's 100 MB per-file limit, so they can be pushed normally.
- Cache folders such as `__pycache__`, `.mpl_config`, and `.yolo_config` are better excluded with a future `.gitignore`.

## Future Improvements

- Add a `.gitignore` for cache files and generated outputs
- Pin exact dependency versions for reproducible model loading
- Add automated tests for GUI-independent pipeline behavior
- Add packaging instructions for distributing the desktop app
