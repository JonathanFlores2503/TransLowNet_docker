import cv2
import os
import shutil
import time
import numpy as np  
import torch 



mainSaveFrames_Path = "FrameStream"
os.makedirs(mainSaveFrames_Path, exist_ok=True)

save_local = True
temporalFrames = 16
frame_count = 0
cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break


    if frame_count == temporalFrames:
        finish_time = time.time()
        print(finish_time - start_time)
        frame_count = 0
        start_time = time.time()


    frame = cv2.resize(frame, (320, 240))
    if save_local:
        frame_path = os.path.join(mainSaveFrames_Path, f"frame_{frame_count:02d}.png")
        cv2.imwrite(frame_path, frame)

    frame_count += 1



cap.release()