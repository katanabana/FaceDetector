from face_detector.face_detector import FaceDetector


def main():
    video_path = "video.mp4"
    verification_image = "face.png"
    output_directory = "result"

    face_detector = FaceDetector(video_path, verification_image, callback=print)

    scenes = face_detector.relevant_scenes(
        tolerance=0.7, frequency=10, quality=0.5
    )  # here you can adjust different parameters of processing

    face_detector.write(scenes, output_directory)


if __name__ == "__main__":
    main()
