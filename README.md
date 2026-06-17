# Air Draw

Air Draw is a computer vision project that allows users to draw in the air using hand gestures instead of a mouse or stylus. The system tracks finger movements in real time and displays the drawing on the screen.

## Features

- Real-time hand tracking
- Draw using finger gestures
- Virtual canvas for sketching
- Simple and interactive user interface
- No physical drawing device required

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## How It Works

1. The webcam captures live video.
2. MediaPipe detects and tracks hand landmarks.
3. The index finger position is used as a virtual pen.
4. Finger movements are converted into drawing strokes on the canvas.

## Installation

Clone the repository:

```bash
git clone https://github.com/sandhiyyaa-v/air-draw.git
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

Run the project:

```bash
python air_draw.py
```

## Future Improvements

- Multiple brush colors
- Eraser functionality
- Save drawings as images
- Gesture-based controls

## Author

Sandhiya Tamizh

ECE Student | Aspiring Software Developer
