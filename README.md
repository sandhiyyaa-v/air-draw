# Air Draw — Gesture-Based Drawing System

Draw on a virtual canvas using only your hand and a webcam. No mouse, no stylus.

## Tech Stack
Python • OpenCV • MediaPipe • NumPy

## Features
- Real-time hand tracking via MediaPipe
- Draw using index finger gestures
- Colour selection from on-screen palette
- Erase, undo, and clear canvas
- Adjustable brush size
- Save artwork as an image

## How It Works
1. Webcam captures live video
2. MediaPipe detects 21 hand landmarks per frame
3. Index finger tip position maps to canvas coordinates
4. Gestures trigger colour/erase/undo actions

## Installation
git clone https://github.com/sandhiyyaa-v/air-draw.git
pip install opencv-python mediapipe numpy
python "air_draw 2.py"

## Author
Sandhiya V — ECE Student, Sathyabama Institute of Science and Technology
