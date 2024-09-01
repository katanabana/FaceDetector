# Face Detector Project

**Face Detector Project** is a Python-based tool that performs face detection in videos and identifies specific scenes where a given face appears.

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
video_path = 'test/30sec.mp4'
verification_image = 'test/face.png'

# Initialize the face detector
face_detector = FaceDetector(video_path, verification_image, callback=print)

# Detect scenes where the face appears and save them
face_detector.write('test/result')
```

## Future Plans

- *UI Development:* We plan to add a graphical user interface to make this tool more accessible to non-developers.
- *Additional Features:* Advanced face matching, add option to detect specific moments instead of scenes, multiple faces detection, etc.