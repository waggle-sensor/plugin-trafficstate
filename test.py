import cv2
import time

video_path='tracking_record1.mov'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_recording.mp4',fourcc, fps, (int(width),int(height)), True)

ret = True
while ret:
    a = time.time()
    ret, frame = cap.read()
    b = time.time()
    out.write(frame)
    c = time.time()
    print(b-a, c-b)

out.release()
cap.release()
