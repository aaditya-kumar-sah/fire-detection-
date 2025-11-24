import cv2
import numpy as np
import threading
from playsound3 import playsound

# ----------------------- SETTINGS -----------------------
cap = cv2.VideoCapture(0)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
alert_sound = 'aag.mp3'
alert_flag = False
cooldown_frames = 100
current_cooldown = cooldown_frames
min_contour_area = 1000

# ----------------------- FUNCTION TO PLAY SOUND -----------------------
def play_alert_sound():
    playsound(alert_sound)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to speed up processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert to HSV and create mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fire_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            fire_detected = True
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Handle alert
    if fire_detected and not alert_flag and current_cooldown == cooldown_frames:
        threading.Thread(target=play_alert_sound, daemon=True).start()
        print("🔥 Fire detected!")
        alert_flag = True

    if alert_flag:
        current_cooldown -= 1
        if current_cooldown <= 0:
            alert_flag = False
            current_cooldown = cooldown_frames

    # Display the frame
    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()