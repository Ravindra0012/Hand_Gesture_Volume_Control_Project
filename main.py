"""
hand_gesture_recognition.py

Hand gesture based system volume control using webcam, Mediapipe, and OpenCV.

Author: Ravindr Singh Nagarkoti
Python Version: 3.11
"""

import time
import subprocess

import cv2  # pylint: disable=import-error
import mediapipe as mp  # pylint: disable=import-error
import numpy as np

# pylint: disable=no-member
# pylint: disable=R0914

def set_volume_mac(percent: int) -> None:
    """
    Set the macOS system volume to a specified percentage.

    Args:
        percent (int): Volume level from 0 to 100.
    """
    percent = int(np.clip(percent, 0, 100))
    script = f"set volume output volume {percent}"
    subprocess.call(["osascript", "-e", script])


def main() -> None:
    """
    Main loop to capture video, detect hand gestures,
    and control system volume based on finger distance.
    """
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    p_time = 0

    min_vol = 0   # Minimum volume (0%)
    max_vol = 100  # Maximum volume (100%)
    vol_bar = 400
    vol = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(
            img,
            f"FPS : {int(fps)}",
            (40, 70),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 0),
            2,
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                height, width, _ = img.shape
                x1, y1 = (
                    int(hand_landmarks.landmark[4].x * width),
                    int(hand_landmarks.landmark[4].y * height),
                )
                x2, y2 = (
                    int(hand_landmarks.landmark[8].x * width),
                    int(hand_landmarks.landmark[8].y * height),
                )
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw landmarks and line between thumb and index finger
                cv2.circle(img, (x1, y1), 14, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 14, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(img, (cx, cy), 14, (255, 0, 255), cv2.FILLED)

                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 8

                # Map the distance to volume range
                vol = np.interp(dist, [3, 47], [min_vol, max_vol])
                vol_bar = np.interp(dist, [3, 47], [400, 150])

                set_volume_mac(vol)

                # Green circle if fingers very close
                if dist < 5:
                    cv2.circle(img, (cx, cy), 20, (0, 255, 0), cv2.FILLED)

        # Volume bar UI
        cv2.rectangle(img, (30, 150), (55, 400), (0, 255, 0), 2)
        cv2.rectangle(img, (30, int(vol_bar)), (55, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            img,
            f"Volume: {int(vol)}%",
            (30, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Hand Volume Control", img)

        if cv2.waitKey(1) & 0xFF == ord("y"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
