import cv2
import pyautogui
import time
from eyetrax import GazeEstimator, run_9_point_calibration


# Settings
SMOOTHING = 0.30
DEAD_ZONE = 20
DOUBLE_BLINK_TIME = 2


# Setup
tracker = GazeEstimator()
run_9_point_calibration(tracker)

screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)

cv2.namedWindow("Eye Tracker", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Eye Tracker", cv2.WND_PROP_TOPMOST, 1)


# Blink tracking
blink_count = 0
last_blink_time = 0
was_blinking = False


# Mouse position
last_x = screen_width // 2
last_y = screen_height // 2

smooth_x = last_x
smooth_y = last_y


while True:

    # Get webcam frame
    success, frame = camera.read()

    if not success:
        break


    # Detect eyes and blinking
    features, blink = tracker.extract_features(frame)

    if features is not None:

        # Move mouse based on gaze
        if not blink:
            x, y = tracker.predict([features])[0]

            x = int(x)
            y = int(y)

            x = max(0, min(x, screen_width - 1))
            y = max(0, min(y, screen_height - 1))

            smooth_x += (x - smooth_x) * SMOOTHING
            smooth_y += (y - smooth_y) * SMOOTHING

            distance = ((smooth_x - last_x) ** 2 + (smooth_y - last_y) ** 2) ** 0.5

            if distance > DEAD_ZONE:
                pyautogui.moveTo(int(smooth_x), int(smooth_y))

                last_x = smooth_x
                last_y = smooth_y


        # Detect a new blink
        if blink and not was_blinking:

            current_time = time.time()

            # Reset blink count if too much time has passed
            if current_time - last_blink_time > DOUBLE_BLINK_TIME:
                blink_count = 0

            blink_count += 1
            last_blink_time = current_time


            # Two blinks = left click
            if blink_count == 2:
                pyautogui.click()
                blink_count = 0


        # Remember current blink state
        was_blinking = blink


    # Show webcam
    cv2.imshow("Eye Tracker", frame)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
camera.release()
cv2.destroyAllWindows()
