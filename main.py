import cv2
import mediapipe as mp
import numpy as np
import time
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
drawingUtils = mp.solutions.drawing_utils
pTime = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400



while (True):
    success, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS : {int(fps)}',(40,70),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for h in result.multi_hand_landmarks:
            hh, w, c = img.shape
            X1, Y1 = int(h.landmark[4].x*w), int(h.landmark[4].y*hh)
            X2, Y2 = int(h.landmark[8].x * w), int(h.landmark[8].y * hh)
            cx, cy = (X1 + X2) // 2, (Y1 + Y2) // 2

            cv2.circle(img, (X1, Y1), 14, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (X2, Y2), 14, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (X1, Y1),(X2, Y2), (255, 0, 0), 2)
            cv2.circle(img, (cx, cy), 14, (255, 0, 255), cv2.FILLED)
            dist = ((X2 - X1) ** 2 + (Y2 - Y1) ** 2) ** 0.5 // 4
            print(dist)

            # Hand range 4 - 60
            # vol range -65 0

            vol = np.interp(dist,[3,47],[minVol,maxVol])
            volBar = np.interp(dist, [3, 47], [400, 150])
            print(vol)
            volume.SetMasterVolumeLevel(vol, None)
            if dist < 5:
                cv2.circle(img, (cx, cy), 20, (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img,(30,150),(55,400),(0,255,0),2)
    cv2.rectangle(img, (30, int(volBar)), (55, 400), (0, 255, 0), cv2.FILLED)
    cv2.imshow('webcam', img)






    key = cv2.waitKey(1)
    if key == ord('y'):
        break

cap.release()
cv2.destroyAllWindows()

