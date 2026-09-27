import cv2
import pyautogui
import time
import speech_recognition as sr
import threading
from eyetrax import GazeEstimator, run_9_point_calibration


# Settings
SMOOTHING = 0.30
DEAD_ZONE = 20


# Setup
tracker = GazeEstimator()
run_9_point_calibration(tracker)

screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)

cv2.namedWindow("Eye Tracker", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Eye Tracker", cv2.WND_PROP_TOPMOST, 1)


# Mouse position
last_x = screen_width // 2
last_y = screen_height // 2

smooth_x = last_x
smooth_y = last_y


recognizer = sr.Recognizer()
microphone = sr.Microphone()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)


def speech_listener():
    while True:
        try:
            with microphone as source:
                audio = recognizer.listen(source)

            text = recognizer.recognize_google(audio)

            if text == "stop":
                break
            elif text == "backspace":
                pyautogui.press("backspace")
            elif text == "enter":
                pyautogui.press("enter")
            elif text == "period":
                pyautogui.write(".")
            elif text == "comma":
                pyautogui.write(",")
            elif text == "click":
                print("click")
                pyautogui.click()
            else:
                pyautogui.write(text + " ")

        except:
            pass

voice_thread = threading.Thread(
    target=speech_listener,
    daemon=True
)

voice_thread.start()


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
                #pyautogui.moveTo(int(smooth_x), int(smooth_y))

                last_x = smooth_x
                last_y = smooth_y


    # Show webcam
    cv2.imshow("Eye Tracker", frame)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
camera.release()
cv2.destroyAllWindows()
