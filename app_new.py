from __future__ import print_function

import numpy as np
import time
import argparse
import cv2

from utils_trt.utils import BaseEngine
from app_utils import RegionOfInterest

from sort import *

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

class RunClass():
    def __init__(self, fps, labels):
        self.flow = []
        self.occupancy_area = 0
        self.density_frames = 0
        self.speed = {}
        self.fps = fps
        self.class_names = load_class_names(labels)

    def set_roi(self, roi):
        self.roi = roi

    def clean_up(self):
        pass

    def calculate_flow(self, t, b, r, l, id_num):
        # Add id_num if the box (t, b, r, l) is entering the ROI
        ret = False
        if self.roi.loi_intersects(t, b, r, l):
            ret = True
            if id_num not in self.flow:
                self.flow.append(id_num)
        return ret

    def get_flow(self):
        return len(self.flow)

    def calculate_density(self, t, b, r, l, outclass):
        # Calculate occupancy area of the class and
        # accumulate the area
        name = self.class_names[outclass]
        #print(f'{name} recognized in {t}, {b}, {r}, {l}')
        if self.roi.contains_center_of_mass(t, b, r, l):
            #print(f'{name} is contained inside the ROI')
            if 'car' in name:
                self.occupancy_area += 4.5 * 1.7
            elif 'bus' in name:
                self.occupancy_area += 13 * 2.55
            elif 'truck' in name:
                 self.occupancy_area += 5.5 * 2
        #self.density_frames += 1

    def get_occupancy(self):
        # Return the accumulated occupied area divided by the road area
        # and frames used for accumulating the occupied area
        print(f'area: {self.occupancy_area} road_area: {self.roi.road_area} density_frames: {self.density_frames}')
        return self.occupancy_area / self.roi.road_area / self.fps

    def calculate_speed(self, t, b, r, l, id_num):
        # Start counting frames when a vehicle touches LoI until it is within the RoI
        # When the vehicle leaves the RoI mark it (-, negative sign) to calculate the speed later
        if self.roi.loi_intersects(t, b, r, l) and self.roi.contains_center_of_mass(t, b, r, l):
            if id_num not in self.speed:
                self.speed[id_num] = 1
            else:
                self.speed[id_num] += 1
        elif self.roi.contains_center_of_mass(t, b, r, l):
            if id_num in self.speed:
                self.speed[id_num] += 1
        else:
            if id_num in self.speed and self.speed[id_num] > 0:
                self.speed[id_num] *= -1

    def get_averaged_speed(self):
        sum_speed = 0.
        for vehicle_id, counted_frames in self.speed.items():
            # negative frames mean the vehicle exited the ROI
            # so considerable to calculate the speed because it means
            # the vehicle traveled the distance of the ROI within the counted frames
            if counted_frames < 0:
                delta_t = -1 * counted_frames / self.fps
                delta_d = self.roi.road_length
                sum_speed += delta_d / delta_t * 3.6 # m/s to km/h
        return 0. if len(self.speed.keys()) == 0 else sum_speed / len(self.speed.keys())

    def reset_flow_and_occupancy(self):
        self.flow = []
        self.occupancy_area = 0.
        self.density_frames = 0

    def reset_speed(self):
        self.speed = {}

    def run(self, trackers, frame):
        for track in trackers:
            print(track)
            id_num = track[4] #Get the ID for the particular track.
            l = track[0]  ## x1
            t = track[1]  ## y1
            r = track[2]-track[0]  ## x2
            b = track[3]-track[1]  ## y2

            name = self.class_names[int(track[-1])]
            frame = cv2.putText(frame, f'{id_num}:{name}', (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            if self.calculate_flow(t, b, r, l, id_num):
                frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)
            else:
                frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)

            self.calculate_density(t, b, r, l, track.outclass)
            self.calculate_speed(t, b, r, l, id_num)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def get_region_of_interest(width, height, roi_name, roi_coordinates, roi_area, roi_length, loi_coordinates):
    try:
        coordinates = []
        loi = []
        for c in loi_coordinates.strip().split(' '):
            x, y = c.split(',')
            loi.append((float(x), float(y)))
        for c in roi_coordinates.strip().split(' '):
            x, y = c.split(',')
            coordinates.append((float(x), float(y)))
        roi = RegionOfInterest(
            coordinates,
            width,
            height,
            roi_area,
            roi_length,
            loi)
        return True, roi
    except Exception as ex:
        return False, str(ex)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    #parser.add_argument("--seq_path", help="Path to detections.", required=True, type=str, default='data')
    #parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)

    # Data
    parser.add_argument("--input-video", type=str, required=True, help="path to dataset")
    parser.add_argument('--labels', dest='labels',
                        action='store', default='coco.names', type=str,
                        help='Labels for detection')
    parser.add_argument('--conf-thresh', type=float, default=0.4)
    return parser.parse_args()


class Predictor(BaseEngine):
    def __init__(self, engine_path , imgsz=(640,640)):
        super(Predictor, self).__init__(engine_path)
        self.imgsz = imgsz # your model infer image size
        self.n_classes = 80  # your model classes

if __name__ == '__main__':

    print(time.time())
    args = parse_args()

    cap = cv2.VideoCapture(args.input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    height, width, c = frame.shape

    pred = Predictor(engine_path=args.engine)

    out_file = open('stats.txt', 'a', buffering=1)
    display = args.display
    total_time = 0.0
    total_frames = 0
    mot_tracker = Sort(max_age=args.max_age,
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    r_class = RunClass(fps, args.labels)

    print('Configuring target area...')
    ret, roi = get_region_of_interest(
        width=width,
        height=height,
        roi_name=args.roi_name,
        roi_coordinates=args.roi_coordinates,
        roi_area=args.roi_area,
        roi_length=args.roi_length,
        loi_coordinates=args.loi_coordinates
    )
    if ret == False:
        print(f'Could not configure region of interest: {roi}. Exiting...')
        exit(1)
    r_class.set_roi(roi)
    print(f'Boundary of the ROI: {roi.roi.bounds}')


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('result.mp4', fourcc, fps, (int(w), int(h)), True)

    print(time.time())
    c = 0
    while True:
        c += 1
        print(c)
        print('s', time.time())

        ret, frame = cap.read()
        if ret == False:
            print('no video frame')
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pred.inference(frame, conf=0.25, end2end=False)

        results = np.asarray(results)
        results[:, 2:4] += results[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        dets = results
        trackers = mot_tracker.update(dets)

        new_frame = r_class.run(trackers, frame)
    '''
        out.write(frame)
    out.release()
    '''
