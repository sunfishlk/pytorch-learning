import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

video_path = './face-recognize/video/3.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = r'./face-recognize/video/output.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

frame_count = 0
max_frames = 100  # 设定一个最大帧数，超过这个数则退出


while True:
    ret, frame = cap.read()
    if not ret or frame_count > max_frames:
        break

    faces = app.get(frame)
    
    for face in faces:
        face.bbox = face.bbox.astype(int)
        face.kps = face.kps.astype(int)
    frame = app.draw_on(frame, faces)
    
    out.write(frame)
    frame_count += 1

cap.release()
out.release()