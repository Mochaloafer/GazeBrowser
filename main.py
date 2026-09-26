import cv2
import pyautogui
import time
from eyetrax import GazeEstimator, run_9_point_calibration

tracker = GazeEstimator()

# calibration
run_9_point_calibration(tracker)

screen_width, screen_height = pyautogui.size()

camera = cv2.VideoCapture(0)


blink_count = 0
last_blink_time = 0
was_blinking = False


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

            pyautogui.moveTo(x, y)

        # make sure click only when blink -> open -> blink
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