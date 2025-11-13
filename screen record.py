import cv2
import numpy as np
import mss
import time

# pip install opencv-python numpy pyautogui mss

# Output file
output_file = "screen_record.mp4"
fps = 20.0
duration = 10  # seconds to record

# Screen dimensions
with mss.mss() as sct:
    monitor = sct.monitors[1]
    width, height = monitor["width"], monitor["height"]

# Define video codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

print("🎥 Recording screen...")
start_time = time.time()

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        out.write(frame)

        if (time.time() - start_time) > duration:
            break

out.release()
print(f"✅ Saved recording to {output_file}")
