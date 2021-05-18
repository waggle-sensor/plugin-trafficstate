import cv2
import numpy as np

from deep_sort.deepsort import *

import torch
# import torchvision

import time

def call_deepsort(args, use_cuda=True):
    m_deepsort = torch.load(args.deepsort_checkpoint)

    if use_cuda:
        m_deepsort.cuda().eval()
    else:
        m_deepsort.eval()

    return m_deepsort


class run_class():
    def __init__(self, DSort, road=20*3*3, fps=60, road_length=20):
        self.DSort = DSort

        self.total_outgoing = []
        self.total_incoming = []
        self.outgoing = []
        self.incoming = []

        self.total_out_occupied = 0
        self.total_in_occupied = 0
        self.out_occupied = 0
        self.in_occupied = 0
        self.road = road

        self.speed_outgoing = {}
        self.speed_incoming = {}
        self.fps = fps
        self.d = road_length
        self.out_speed = {}
        self.in_speed = {}


    def flow(self, t, b, r, l, id_num):
        if t < 540 and b > 540:     ### if a box acrosses the bottom (yellow) line
            if r < 540:             ### and the box on the left side (enter RoI)
                if id_num not in self.outgoing:
                    self.outgoing.append(id_num)
            elif l > 670:           ### and the box on the right side (exit RoI)
                if id_num not in self.incoming:
                    self.incoming.append(id_num)



    def density(self, t, b, r, l, outclass):
        if t < 540 and b > 340:
            if r < 540:
                if outclass == 2:  ### car
                    self.out_occupied += 4.5*1.7
                elif outclass == 5:  ### bus
                    self.out_occupied += 13*2.55
                elif outclass == 7:  ### truck
                    self.out_occupied += 5.5*2

            elif l > 670:
                if outclass == 2:  ### car
                    self.in_occupied += 4.5*1.7
                elif outclass == 5:  ### bus
                    self.in_occupied += 13*2.55
                elif outclass == 7:  ### truck
                    self.in_occupied += 5.5*2


    def speed(self, t, b, r, l, id_num):
        if t < 540 and b > 540:     ### yellow line
            if r < 540:             ### come into yellow line
                if id_num not in self.speed_outgoing:
                    self.speed_outgoing[id_num] = [1]    ### the vehicle is getting further, and firstly captured
                else:
                    self.speed_outgoing[id_num][0] += 1   ### the vehicle is gettig further, and captured multiple times
            elif l > 670:           ### get out from yellow line
                if id_num not in self.speed_incoming:   ### the vehicle is getting closer, and don't know when it first get into the RoI
                    pass
                elif len(self.speed_incoming[id_num]) < 2:   ### the vehicle is getting closer, and now exiting the RoI -- need to calculate speed
                    self.speed_incoming[id_num][0] += 1
                    self.speed_incoming[id_num].append(1)

        if t < 340 and b > 340:     ### blue line
            if r < 540:             ### get out from blue line
                if id_num not in self.speed_outgoing:    ### the vehicle is getting further, and don't know when it firstly get into the RoI
                    pass
                elif len(self.speed_outgoing[id_num]) < 2:   ### the vehicle is getting further, and now exiting the RoI
                    self.speed_outgoing[id_num][0] += 1
                    self.speed_outgoing[id_num].append(1)
            elif l > 670:           ### come into blue line
                if id_num not in self.speed_incoming:
                    self.speed_incoming[id_num] = [1]   ### the vehicle is getting closer, and firstly captured
                else:
                    self.speed_incoming[id_num][0] += 1   ### the vehicle is getting closer, and cpatured multiple times


        if b < 340 and r < 540:       #### outgoing -- the vehicle exited the RoI -- calculate speed
            if id_num in self.speed_outgoing:
#                 print(self.speed_outgoing)
                delta_t = self.speed_outgoing[id_num][0] * (1/self.fps)
                delta_d = self.d
                self.out_speed[id_num] = round((delta_d/delta_t)*3.6, 2)  ### change unit from m/s to km/h by multiplying 3.6
#                 print('>> outgoing speed: ', self.out_speed[id_num], 'km/h')
                self.speed_outgoing.pop(id_num)
        if t > 540 and l > 670:        #### incoming -- the vehicle exited the RoI -- calculate speedl
            if id_num in self.speed_incoming:
#                 print(self.speed_incoming)
                delta_t = self.speed_incoming[id_num][0] * (1/self.fps)
                delta_d = self.d
                self.in_speed[id_num] = round((delta_d/delta_t)*3.6, 2)  ### change unit from m/s to km/h by multiplying 3.6
#                 print('>>> incoming speed: ', self.in_speed[id_num], 'km/h')
                self.speed_incoming.pop(id_num)




    def processing(self, boxes, class_names, frame, ret):
        if ret == False:
            print(ret, 'no_frame')
            return
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.uint8)

            tracker, detections_class = self.DSort.a_run_deep_sort(frame, boxes)

            for track in tracker.tracks:
#                 print('track.is_confirmed(): ', track.is_confirmed())
#                 print('track.time_since_update: ', track.time_since_update)
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr() #Get the corrected/predicted bounding box
                id_num = str(track.track_id) #Get the ID for the particular track.
                features = track.features #Get the feature vector corresponding to the detection.

                l = bbox[0]  ## x1
                t = bbox[1]  ## y1
                r = bbox[2]  ## x2
                b = bbox[3]  ## y2

                self.flow(t, b, r, l, id_num)
                self.density(t, b, r, l, track.outclass)
                self.speed(t, b, r, l, id_num)



    def run_dsort(self, boxes, class_names, frame, ret):
        self.processing(boxes, class_names, frame, ret)


if __name__=='__main__':
    video_path = '../tracking_record1.mov'
    class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
                   'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']



    cap = cv2.VideoCapture(video_path)
    cvfps = cap.get(cv2.CAP_PROP_FPS)
    cvfps = 30
    print('fps:  ', cvfps)

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    m_deepsort = call_deepsort()
    DSort = deepsort_rbc(m_deepsort, width, height)
    RClass = run_class(DSort, road=20*3*3, fps=cvfps, road_length=20)

    total_frames = 0.
    ret = True
    while ret:
        total_frames += 1.
        ret, frame = cap.read()
        if total_frames % 2 == 1:
            continue
        RClass.run_dsort(frame, class_names, frame, ret)

#         print('a: ', round(RClass.out_occupied/RClass.road*100, 2))
#         print('b: ', round(RClass.in_occupied/RClass.road*100, 2))

#         RClass.out_occupied = 0
#         RClass.in_occupied = 0




    ### Split results (averaged in sec -- or total length of the video)
    ##### traffic flow
    total_sec = total_frames/cvfps
    print(total_frames, cvfps, total_sec)
    print(len(RClass.outgoing)/total_sec, len(RClass.incoming)/total_sec)
    RClass.outgoing = []
    RClass.incoming = []

    ##### traffic density
    print('a: ', round(RClass.out_occupied/(RClass.road*total_frames)*100, 2))
    print('b: ', round(RClass.in_occupied/(RClass.road*total_frames)*100, 2))
    RClass.out_occupied = 0
    RClass.in_occupied = 0

    ##### traffic speed
    s = [0,0,0,0]
    for k, v in RClass.out_speed.items():
        s[0] += 1
        s[1] += v
    for k, v in RClass.in_speed.items():
        s[2] += 1
        s[3] += v
    print(s)
    print('speed out: ', round(s[1]/s[0], 2))
    print('speed in: ', round(s[3]/s[2], 2))

    print('stop plugin')
    cap.release()


