from yolov4 import *
from deepsort import *

import numpy as np
import cv2
import time
from time import sleep
from multiprocessing import Process, Queue, Value

import waggle.plugin as plugin

TOPIC_INPUT_IMAGE = "road_image"
TOPIC_CLOUDCOVER = "env.state.traffic"

plugin.init()

def run_videocapture(video_queue):
    print('hi')
    designated_fps = 20.0
    cap = cv2.VideoCapture(0)

    """Define the codec and create VideoWriter object"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outname = time.time() + '.avi'
    out = cv2.VideoWriter(outname,fourcc, designated_fps, (1194,670))
    video_queue.put(outname)

    print(cap.isOpened())
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        count += 1
        print(count)
        if ret == True:
            frame = cv2.flip(frame,0)

            """write the flipped frame"""
            out.write(frame)

        if count == 5*designated_fps:  """if the video has been recorded for 5 sec"""
            break

    """lelease everything if job is finished"""
    cap.release()
    out.release()


def run_yolov4(kill_queue, output_queue, video_queue):
    video_path='../tracking_record1.mov'
    cap = cv2.VideoCapture(video_path)

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    m = call_yolov4()
    ODetect = yolov4_trck(m)

    namesfile='detection/coco.names'
    num_classes = m.num_classes
    class_names = load_class_names(namesfile)

    while kill_queue.empty():
        if output_queue.full():
            sleep(0.1)
            continue

        ret, frame = cap.read()
        if ret == False:
            break

        output_queue.put((ODetect.run_yolov4(frame, ret), frame, ret))

    kill_queue.put(1)
    cap.release()

def run_deepsort(kill_queue, input_queue):
    video_path='../tracking_record1.mov'

    cap = cv2.VideoCapture(video_path)
    cvfps = cap.get(cv2.CAP_PROP_FPS)
    cvfps = 10

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    cap.release()


    namesfile='detection/coco.names'
    class_names = load_class_names(namesfile)

    """deepsort model"""
    m_deepsort = call_deepsort()
    DSort = deepsort_rbc(m_deepsort, width, height)
    RClass = run_class(DSort, road=20*3*3, fps=cvfps, road_length=20)

    total_frames = 0
    while kill_queue.empty():
        if input_queue.empty():
            sleep(0.1)
            continue
        total_frames += 1
        if total_frames == 8*cvfps:
            kill_queue.put(2)

        detection, frame, ret = input_queue.get()
        RClass.run_dsort(detection, class_names, frame, ret)

        if total_frames % cvfps == 0:
            plugin.publish('env.traffic_flow.outgoing', len(RClass.outgoing))
            plugin.publish('env.traffic_flow.incoming', len(RClass.incoming))
            RClass.total_outgoing = RClass.total_outgoing + RClass.outgoing
            RClass.total_incoming = RClass.total_incoming + RClass.incoming
            RClass.outgoing = []
            RClass.incoming = []

            plugin.publish('env.traffic_density.outgoing', round(RClass.out_occupied/(RClass.road*cvfps)*100, 2))
            plugin.publish('env.traffic_density.incoming', round(RClass.in_occupied/(RClass.road*cvfps)*100, 2))
            RClass.total_out_occupied = RClass.total_out_occupied + RClass.out_occupied
            RClass.total_in_occupied = RClass.total_in_occupied + RClass.in_occupied
            RClass.out_occupied = 0
            RClass.in_occupied = 0



    """Split results (averaged in sec -- or total length of the video)"""
    """traffic flow"""
    total_sec = total_frames/cvfps
    plugin.publish('env.traffic_flow.outgoing', len(RClass.outgoing))
    plugin.publish('env.traffic_flow.incoming', len(RClass.incoming))


    """traffic density"""
    plugin.publish('env.traffic_density.outgoing', round(RClass.out_occupied/(RClass.road*cvfps)*100, 2))
    plugin.publish('env.traffic_density.incoming', round(RClass.in_occupied/(RClass.road*cvfps)*100, 2))


    """traffic speed"""
    s = [0,0,0,0]
    for k, v in RClass.out_speed.items():
        s[0] += 1
        s[1] += v
    for k, v in RClass.in_speed.items():
        s[2] += 1
        s[3] += v

    speed_stack['raw'] = s
    if s[0] != 0:
        plugin.publish('env.traffic_speed.outgoing', round(s[1]/s[0], 2))
    else:
        plugin.publish('env.traffic_speed.outgoing', 'n/a')
    if s[2] != 0:
        plugin.publish('env.traffic_speed.incoming', round(s[3]/s[2], 2))
    else:
        plugin.publish('env.traffic_speed.outgoing', 'n/a')

    print('stop plugin')



if __name__ == '__main__':
    video_queue = Queue(maxsize=1)
    message_queue = Queue(maxsize=1)
    kill_queue = Queue(maxsize=1)

    yolo_process = Process(target=run_yolov4, args=(kill_queue, message_queue, video_queue,))
    dsort_process = Process(target=run_deepsort, args=(kill_queue, message_queue,))

    yolo_process.start()
    dsort_process.start()

    try:
        while kill_queue.empty():
            sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print('>>>>>> video fps must higher or equal than 12 <<<<<<')
        if kill_queue.empty():
            kill_queue.put(0)
        if yolo_process.is_alive():
            yolo_process.terminate()
        if dsort_process.is_alive():
            dsort_process.terminate()

