import cv2
from eyetrax import GazeEstimator, run_9_point_calibration

tracker = GazeEstimator()

# Calibrate
run_9_point_calibration(tracker)

# Start camera
camera = cv2.VideoCapture(0)

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        features, blink = tracker.extract_features(frame)

        if features is not None and not blink:
            x, y = tracker.predict([features])[0]
            print("Looking at:", int(x), int(y))

        cv2.imshow("Eye Tracker", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
