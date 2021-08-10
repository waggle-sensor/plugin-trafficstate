import cv2
import numpy as np
import time

from tool.utils import *
from tool.torch_utils import do_detect
from tool.darknet2pytorch import Darknet

from deep_sort.deepsort import *
import torch

import argparse


class yolov4_trck():
    def __init__(self, use_cuda, cfgfile='yolov4.cfg', weightfile='yolov4.weights'):
        self.m = Darknet(cfgfile)
        self.m.load_weights(weightfile)

        if use_cuda:
            self.m.cuda().eval()
        else:
            self.m.eval()

        self.use_cuda = use_cuda

    def run_yolov4(self, frame, ret):
        if ret == False:
            print(ret, 'no_frame')
            return
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sized = cv2.resize(frame, (512, 512))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            #### Start detection using do_detect() function
    #             start = time.time()
            ######### output must be boxes[0], which contains tbrl, confidence level, and class number
            boxes = do_detect(self.m, sized, 0.4, 0.6, self.use_cuda)
    #             print(type(boxes), len(boxes[0]), ': number of detected cars', boxes[0])
    #             finish = time.time()
    #             print('yolo elapsed in: %f sec' % (finish - start))
            return boxes[0]


def call_deepsort(use_cuda, wt_path='model640.pt'):
    if use_cuda:
        m_deepsort = torch.load(wt_path)
        m_deepsort.cuda().eval()
    else:
        m_deepsort = torch.load(wt_path, map_location=torch.device('cpu'))
        m_deepsort.eval()

    return m_deepsort


class run_class():
    def __init__(self, DSort, road, fps, road_length):
        self.DSort = DSort

        self.outgoing = []
        self.incoming = []

        self.out_occupancy_area = 0
        self.in_occupancy_area = 0
        self.out_occupancy = 0
        self.in_occupancy = 0

        self.speed_outgoing = {}
        self.speed_incoming = {}
        self.fps = fps
        self.out_speed = {}
        self.in_speed = {}

        self.d = road_length
        self.road = road


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
                    self.out_occupancy_area += 4.5*1.7
                    self.out_occupancy += 4.5
                elif outclass == 5:  ### bus
                    self.out_occupancy_area += 13*2.55
                    self.out_occupancy += 13
                elif outclass == 7:  ### truck
                    self.out_occupancy_area += 5.5*2
                    self.out_occupancy += 5.5

            elif l > 670:
                if outclass == 2:  ### car
                    self.in_occupancy_area += 4.5*1.7
                    self.in_occupancy += 4.5
                elif outclass == 5:  ### bus
                    self.in_occupancy_area += 13*2.55
                    self.in_occupancy += 13
                elif outclass == 7:  ### truck
                    self.in_occupancy_area += 5.5*2
                    self.in_occupancy += 5.5


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




    def run_dsort(self, boxes, class_names, frame, ret):
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


def run(ODetect, Rclass, cvfps):
    test = False
    total_frames = 0
    while True:
        if total_frames == 60*6:
            break
        total_frames += 1

        ret, frame = cap.read()
        if ret == False:
            break

        result = ODetect.run_yolov4(frame, ret)
        RClass.run_dsort(result, class_names, frame, ret)

        if total_frames % cvfps == 0:
            ##### traffic occupancy
            print('road occupancy', RClass.out_occupancy/cvfps, 'm', RClass.in_occupancy/cvfps, 'm')
            print('road occupancy', RClass.out_occupancy_area/RClass.d/cvfps, RClass.in_occupancy_area/RClass.d/cvfps)
            RClass.out_occupancy_area = 0
            RClass.in_occupancy_area = 0
            RClass.out_occupancy = 0
            RClass.in_occupancy = 0

            ##### traffic flow
            print('traffic flow', len(RClass.outgoing), len(RClass.incoming))
            RClass.outgoing = []
            RClass.incoming = []

    ##### traffic speed
    s = [0,0,0,0]
    for k, v in RClass.out_speed.items():
        s[0] += 1
        s[1] += v
    for k, v in RClass.in_speed.items():
        s[2] += 1
        s[3] += v
    print(s)

    if s[0] == 0:
        print('speed out: ', 0, 'm/s')
    else:
        print('speed out: ', round(s[1]/s[0], 2), 'm/s')
        print('speed out: ', round(s[1]/s[0], 2)/3.6, 'km/h')

    if s[2] == 0:
        print('speed in: ', 0, 'm/s')
    else:
        print('speed in: ', round(s[3]/s[2], 2), 'm/s')
        print('speed in: ', round(s[3]/s[2], 2)/3.6, 'km/h')

    print('stop plugin')
    cap.release()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_false', default=True)
    args = parser.parse_args()


    video_path='tracking_record1.mov'
    usecuda = False
    roadlength = 60*3
    roadarea = 60*3*3

    cap = cv2.VideoCapture(video_path)
    cvfps = cap.get(cv2.CAP_PROP_FPS)
    print('fps:  ', cvfps)

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    ### ML models
    #### yolo model
    ODetect = yolov4_trck(use_cuda=usecuda)

    namesfile='detection/coco.names'
    num_classes = ODetect.m.num_classes
    class_names = load_class_names(namesfile)

    #### deepsort model
    m_deepsort = call_deepsort(use_cuda=usecuda)
    DSort = deepsort_rbc(m_deepsort, width, height, use_cuda=usecuda)
    RClass = run_class(DSort, road=roadarea, fps=cvfps, road_length=roadlength)

    run(ODetect, RClass, cvfps)
