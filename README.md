# Face Detector

**Face Detector** is a Python-based tool that performs face detection in videos and identifies specific scenes where a
given face appears.

## Features

- Detect faces in video frames using a reference image.
- Extract specific scenes from a video where the given face is detected.
- Automatically generate new video files with detected scenes.
- Designed for future extension with a UI.

## Installation

### Prerequisites

- **Python 3.8+**
- **ffmpeg** installed on your system:
    - **For Ubuntu/Debian**: `sudo apt install ffmpeg`
    - **For macOS**: `brew install ffmpeg`
    - **For Windows**: Download and install from [ffmpeg.org](https://ffmpeg.org/download.html).

### Clone the Repository

```bash
git clone https://github.com/katanabana/face-detector-project.git
cd face-detector-project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Here's an example of how to use the FaceDetector class in your own script:

```python
from face_detector.face_detector import FaceDetector

video_path = 'video.mp4'
verification_image = 'face.png'
output_directory = 'result'

face_detector = FaceDetector(video_path, verification_image, callback=print)

# Adjust different parameters of processing
scenes = face_detector.relevant_scenes(tolerance=0.7, frequency=10, quality=0.5)

face_detector.write(scenes, output_directory)
```

## Future Plans

- *UI Development:* We plan to add a graphical user interface to make this tool more accessible to non-developers.
- *Additional Features:* Advanced face matching, add option to detect specific moments instead of scenes, multiple faces
  detection, etc.
