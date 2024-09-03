import re
import subprocess

import cv2
import numpy as np
from numpy.linalg import norm
import os
from PIL import Image
import face_recognition
from scenedetect import open_video
from scenedetect.detectors import ContentDetector
from face_detector.custom_scene_manager import CustomSceneManager


class InvalidFacesNumber(Exception):
    def __init__(self, number_of_faces):
        self.number_of_faces = number_of_faces
        message = f"Verification image should be an image with exactly 1 face. " \
                  "Provided image contains {number_of_faces} faces."
        super().__init__(message)


class FilesAlreadyExist(Exception):
    def __init__(self, directory, file_names):
        names = ',\n'.join(['"' + name + '"' for name in file_names])
        super().__init__(f'The directory "{directory}" already contains:\n{names}')


class FaceDetector:

    def __init__(self, video, verification_image, callback=lambda current, total: None):
        # Load the verification image and convert it to RGB format
        img = Image.open(verification_image)
        img = img.convert('RGB')
        img_encoding = np.array(img)

        # Detect face encodings in the verification image
        faces = face_recognition.face_encodings(img_encoding, model='large')

        # Ensure the verification image contains exactly one face
        if len(faces) != 1:
            raise InvalidFacesNumber(len(faces))

        # Store the face encoding for later comparison
        self.verification_face = faces[0]

        # Open the video file
        self.video_path = video
        self.video_cap = cv2.VideoCapture(video)

        # Set up callback for progress tracking
        self.callback = callback

        # Get video properties
        self.fps = int(self.video_cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

    def scenes(self):
        video = open_video(self.video_path)
        scene_manager = CustomSceneManager()
        detector = ContentDetector()
        scene_manager.add_detector(detector)
        start_frame = end_frame = 0
        fps = self.fps
        for end_frame in scene_manager.custom_detect_scenes(video):
            yield start_frame / fps, end_frame / fps
            self.callback(end_frame, self.frame_count)
            start_frame = end_frame
        if end_frame:
            yield start_frame / fps, self.frame_count / fps
            self.callback(self.frame_count, self.frame_count)

    def relevant_scenes(self, tolerance=0.7, frequency=10, quality=0.5):
        # Iterate over detected scenes in the video
        for i, (start, end) in enumerate(self.scenes()):
            # Calculate the start and end frames
            start_frame = int(start * self.fps)
            end_frame = int(end * self.fps)

            # Process each frame within the scene
            frame_count = end_frame - start_frame
            for frame_idx in range(start_frame, end_frame, frequency):
                # Seek to the frame specified by frame_idx
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                # Read the frame
                ret, frame = self.video_cap.read()
                if not ret:
                    break

                # Resize the frame to reduce processing load
                frame_resized = cv2.resize(frame, (0, 0), fx=quality, fy=quality)

                # Detect faces in the frame
                face_encodings = face_recognition.face_encodings(frame_resized, model='large')

                # Compare detected faces with the verification face
                if face_encodings:
                    differences = norm(face_encodings - self.verification_face, axis=1)
                    if any(differences <= tolerance):
                        yield start, end
                        break  # Move to the next scene

    def write(self, scenes, directory):
        # Check if the output directory exists
        if os.path.exists(directory):
            # Check if the directory already contains relevant scene files

            # Define a regular expression pattern to match the filenames
            pattern = re.compile(r'^scene_\d+(_(temp|audio))?\.mp4$')

            # List all files in the directory and filter those that match the pattern
            existing_files = [f for f in os.listdir(directory) if pattern.match(f)]

            if existing_files:
                raise FilesAlreadyExist(directory, existing_files)

        # Try to create the directory if it does not exist
        else:
            os.makedirs(directory)

        # Open video capture again to extract and save scenes
        self.video_cap = cv2.VideoCapture(self.video_path)

        # Iterate over detected scenes where the face appears
        for i, (start, end) in enumerate(scenes):
            # Calculate the start and end frames
            start_frame = int(start * self.fps)
            end_frame = int(end * self.fps)

            # Seek to the starting frame of the scene
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Prepare video writer for saving the scene
            video_output_path = os.path.join(directory, f'scene_{i + 1}_temp.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None

            # Write frames to the output file
            for frame_idx in range(start_frame, end_frame):
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                if out is None:
                    height, width, _ = frame.shape
                    out = cv2.VideoWriter(video_output_path, fourcc, self.fps, (width, height))

                out.write(frame)

            if out:
                out.release()

            # Extract audio from the original video for this scene using ffmpeg
            audio_output_path = os.path.join(directory, f'scene_{i + 1}_audio.mp4')
            ffmpeg_cmd_audio = [
                'ffmpeg', '-i', self.video_path, '-ss', str(start), '-to', str(end),
                '-vn', '-acodec', 'copy', audio_output_path
            ]
            subprocess.run(ffmpeg_cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # Merge the video and audio into a final output file using ffmpeg
            final_output_path = os.path.join(directory, f'scene_{i + 1}.mp4')
            ffmpeg_cmd_merge = [
                'ffmpeg', '-i', video_output_path, '-i', audio_output_path, '-c:v', 'copy',
                '-c:a', 'aac', '-strict', 'experimental', final_output_path
            ]
            subprocess.run(ffmpeg_cmd_merge, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # Clean up temporary video and audio files
            os.remove(video_output_path)
            os.remove(audio_output_path)

        # Release the video capture at the end
        self.video_cap.release()
