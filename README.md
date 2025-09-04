# Hand Gesture Volume Control

Control your system volume using hand gestures detected via your webcam. This project uses **Mediapipe** for hand landmark detection, **OpenCV** for video processing, and system-specific commands to adjust volume.

---

## Features

- Detects thumb and index finger distance using webcam.
- Maps finger distance to system volume.
- Visual volume bar and FPS counter.
- Green indicator when fingers are pinched close.

---

## Requirements

- Python 3.11
- Webcam connected and accessible

---

## Python Dependencies

- opencv-python
- mediapipe
- numpy
- pycaw (Windows only)

---

## Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/hand-gesture-volume-control.git
    cd hand-gesture-volume-control
    ```

2. **Create and activate a Python virtual environment (recommended):**

    ```bash
    python -m venv handtrack_env
    # Activate environment:
    # On macOS/Linux:
    source handtrack_env/bin/activate
    # On Windows:
    handtrack_env\Scripts\activate
    ```

3. **Install dependencies:**

    - For **macOS/Linux**:

      ```bash
      pip install opencv-python mediapipe numpy
      ```

    - For **Windows**:

      ```bash
      pip install opencv-python mediapipe numpy pycaw comtypes
      ```

---

## Running the Script

### macOS

Run:

```bash
python hand_gesture_recognition_mac.py
```
Note: This script uses AppleScript via osascript to control system volume.


### Windows
Run:

```bash

python hand_gesture_recognition_windows.py
```
Note: This script uses the pycaw library to control system volume.

## Usage
```
Allow camera access when prompted.

Show your thumb and index finger in front of the webcam.

Pinch your fingers to decrease volume, move apart to increase.

The volume bar and percentage on screen reflect the current volume.

Press y key to exit the program.
```
## Troubleshooting
```
Ensure camera permissions are granted.

Check that Python dependencies are installed in the active environment.

If volume doesn't go to minimum or maximum, adjust the finger distance ranges in the script to suit your hand size.

On Windows, run the terminal as Administrator if volume control doesn't work.
```
## Author
Ravindr Singh Nagarkoti

Python Version: 3.11  
