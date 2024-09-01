from face_detector.face_detector import FaceDetector


def main():
    video_path = 'test/30sec.mp4'
    verification_image = 'test/face.png'

    # Initialize the face detector
    face_detector = FaceDetector(video_path, verification_image, callback=print)

    # Detect scenes where the face appears and save them
    face_detector.write('test/result')


if __name__ == '__main__':
    main()
