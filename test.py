import cv2

video_path='tracking_record1.mov'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('test_recording.mp4',fourcc, fps, (int(width),int(height)), True)

ret, frame = cap.read()
out.write(frame)

out.release() 
cap.release()