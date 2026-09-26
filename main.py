import cv2
import pyautogui
import time
from eyetrax import GazeEstimator, run_9_point_calibration

tracker = GazeEstimator()

run_9_point_calibration(tracker)

screen_width, screen_height = pyautogui.size()

camera = cv2.VideoCapture(0)

cv2.namedWindow("Eye Tracker", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Eye Tracker",
    cv2.WND_PROP_TOPMOST,
    1
)

blink_count = 0
last_blink_time = 0
was_blinking = False

last_x = screen_width // 2
last_y = screen_height // 2

dead_zone = 30

while True:
    success, frame = camera.read()

    if not success:
        break

    features, blink = tracker.extract_features(frame)

    if features is not None:

        if not blink:
            x, y = tracker.predict([features])[0]
            x = int(x)
            y = int(y)

            distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5

            if distance > dead_zone:
                pyautogui.moveTo(x, y)

                last_x = x
                last_y = y

        if blink and not was_blinking:

            current_time = time.time()

            if current_time - last_blink_time > 2:
                blink_count = 0

            blink_count += 1
            last_blink_time = current_time

            if blink_count == 2:

                pyautogui.click()
                blink_count = 0

        was_blinking = blink

    cv2.imshow("Eye Tracker - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
