from __future__ import print_function

import numpy as np
import time
import argparse
import cv2
import datetime
import os
import ffmpeg

from utils_trt.utils import BaseEngine
from app_utils import RegionOfInterest

from sort import *

from waggle.plugin import Plugin
from waggle.data.vision import Camera
from waggle.data.vision import resolve_device
from waggle.data.timestamp import get_timestamp


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

    def calculate_density(self, t, b, r, l, name):
        # Calculate occupancy area of the class and
        # accumulate the area
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
                frame = cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)\

            self.calculate_density(t, b, r, l, name)
            self.calculate_speed(t, b, r, l, id_num)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


class Predictor(BaseEngine):
    def __init__(self, engine_path , imgsz=(640,640)):
        super(Predictor, self).__init__(engine_path)
        self.imgsz = imgsz # your model infer image size
        self.n_classes = 80  # your model classes


def get_region_of_interest(width, height, roi_coordinates, roi_area, roi_length, loi_coordinates):
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

def get_stream_info(stream_url):
    try:
        input_probe = ffmpeg.probe(stream_url)
        fps = eval(input_probe['streams'][0]['r_frame_rate'])
        width = int(input_probe['streams'][0]['width'])
        height = int(input_probe['streams'][0]['height'])
        return True, fps, width, height
    except:
        return False, 0., 0, 0


"""
    ffmpeg describes that 
    https://trac.ffmpeg.org/wiki/ChangingFrameRate
"""
def take_sample(stream, duration, skip_second, resampling, resampling_fps):
    stream_url = resolve_device(stream)
    # Assume PyWaggle's timestamp is in nano seconds
    timestamp = get_timestamp() + skip_second * 1e9
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = os.getcwd()
    filename_raw = os.path.join(script_dir, 'record_raw.mp4')
    filename = os.path.join(script_dir, 'record.mp4')

    c = ffmpeg.input(stream_url, ss=skip_second).output(
        filename_raw,
        codec = "copy", # use same codecs of the original video
        f='mp4',
        t=duration).overwrite_output()
    print(c.compile())
    c.run(quiet=True)

    d = ffmpeg.input(filename_raw)
    if resampling:
        print(f'Resampling to {resampling_fps}...')
        d = ffmpeg.filter(d, 'fps', fps=resampling_fps)
        d = ffmpeg.output(d, filename, f='mp4', t=duration).overwrite_output()
    else:
        d = ffmpeg.output(d, filename, codec="copy", f='mp4', t=duration).overwrite_output()

    print(d.compile())
    d.run(quiet=True)
    # TODO: We may want to inspect whether the ffmpeg commands succeeded
    return True, filename, timestamp


def run(args):

    with Plugin() as plugin:
        timestamp = int(time.time()*1e9)
        plugin.publish('traffic.state.log', 'Traffic State Estimator: Getting Video', timestamp=timestamp)
        print(f"Getting Video at time: {timestamp}")

        device_url = resolve_device(args.stream)
        ret, fps, width, height = get_stream_info(device_url)
        if ret == False:
            print(f'Error probing {device_url}. Please make sure to put a correct video stream')
            return 1
        print(f'Input stream {device_url} with size of W: {width}, H: {height} at {fps} FPS')

        # If resampling is True, we use resampling_fps for inferencing as well as sampling
        if args.resampling:
            fps = args.resampling_fps
            print(f'Input will be resampled to {args.resampling_fps} FPS')

        timestamp = int(time.time()*1e9)
        plugin.publish('traffic.state.log', 'Traffic State Estimator: Loading Models', timestamp=timestamp)
        print(f"Loading Models at time: {timestamp}")

        pred = Predictor(engine_path=args.engine)
        mot_tracker = Sort(max_age=args.max_age,
                        min_hits=args.min_hits,
                        iou_threshold=args.iou_threshold) #create instance of the SORT tracker
        r_class = RunClass(fps, args.labels)

        print('Configuring target area...')
        ret, roi = get_region_of_interest(
            width=width,
            height=height,
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

        sampling_countdown = -1
        if args.sampling_interval > -1:
            print(f'Input video will be sampled every {args.sampling_interval}th inferencing')
            sampling_countdown = args.sampling_interval

        timestamp = int(time.time()*1e9)
        plugin.publish('traffic.state.log', 'Traffic State Estimator: Starting Estimation', timestamp=timestamp)
        print(f"Starting Estimation at time: {timestamp}")

        print('Starting traffic state estimation..')

        print(f'Grabbing video for {args.duration} seconds')
        ret, filename, timestamp = take_sample(
            stream=args.stream,
            duration=args.duration,
            skip_second=args.skip_second,
            resampling=args.resampling,
            resampling_fps=args.resampling_fps
        )
        timestamp = int(timestamp)
        if ret == False:
            print('Coud not sample video. Exiting...')
            return 1

        print('Analyzing the video...')
        total_frames = 0
        do_sampling = False
        if sampling_countdown > 0:
            sampling_countdown -= 1
        elif sampling_countdown == 0:
            do_sampling = True
            sampling_countdown = args.sampling_interval

        print('do sampling', do_sampling)

        cap = cv2.VideoCapture(filename)
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

        if do_sampling:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter("sample.mp4", fourcc, fps, (int(width), int(height)), True)

        c = 0
        while True:
            ret, frame = cap.read()
            if ret == False:
                print('no video frame')
                break

            c += 1
            print(c)

            results = pred.inference(frame, conf=args.det_thr, end2end=False)

            results = np.asarray(results)
            results[:, 2:4] += results[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            dets = results
            trackers = mot_tracker.update(dets)
            sample = r_class.run(trackers, frame)

            if do_sampling:
                coordinates = r_class.roi.get_coordinates()
                sample = cv2.polylines(sample, coordinates,
                            True, (255, 0, 0), 2)
                coordinates = r_class.roi.get_loi()
                sample = cv2.polylines(sample, coordinates,
                            True, (0, 0, 255), 4)
                out.write(sample)
            total_frames += 1

            if total_frames % fps == 0:
                elapsed_time = timestamp + int((total_frames / fps) * 1e9)
                ##### traffic occupancy
                occupancy = r_class.get_occupancy()
                plugin.publish(
                    'traffic.state.occupancy',
                    occupancy,
                    timestamp=elapsed_time)

                ##### traffic flow
                flow = r_class.get_flow()
                plugin.publish(
                    'traffic.state.flow',
                    flow,
                    timestamp=elapsed_time)

                print(f'{datetime.datetime.fromtimestamp(elapsed_time / 1.e9)} Traffic occupancy: {occupancy} flow: {flow}')
                # Reset the accumulated values
                r_class.reset_flow_and_occupancy()

        ##### traffic speed
        averaged_speed = r_class.get_averaged_speed()
        plugin.publish(
            'traffic.state.averaged_speed',
            averaged_speed,
            timestamp=timestamp)
        print(f'{datetime.datetime.fromtimestamp(timestamp / 1.e9)} Traffic speed: {averaged_speed}')

        if do_sampling:
            out.release()
            plugin.upload_file("sample.mp4")
        r_class.clean_up()
        print('Tracker is cleaned up for next analysis')



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('-engine', type=str, required=True)
    parser.add_argument("-max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=30)
    parser.add_argument("-min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("-iou_threshold", help="Minimum IOU for match.", type=float, default=0.1)

    # Data
    parser.add_argument('-labels', dest='labels',
                        action='store', default='coco.names', type=str,
                        help='Labels for detection')
    parser.add_argument("-detection-threshold", dest='det_thr', type=float, default=0.25)


    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="camera", type=str,
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-duration', dest='duration',
        action='store', default=10., type=float,
        help='Time duration for input video')
    parser.add_argument(
        '-resampling', dest='resampling', default=False,
        action='store_true', help="Resampling the sample to -resample-fps option (defualt 12)")
    parser.add_argument(
        '-resampling-fps', dest='resampling_fps',
        action='store', default=12, type=int,
        help='Frames per second for input video')
    parser.add_argument(
        '-skip-second', dest='skip_second',
        action='store', default=3., type=float,
        help='Seconds to skip before recording')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Inferencing interval for sampling results')


    parser.add_argument(
        '-loi-coordinates', dest='loi_coordinates',
        action='store', type=str, default="0.3,0.3 0.6,0.3",
        help='X,Y Coordinates of Line of interest for flow calculation')
    parser.add_argument(
        '-roi-area', dest='roi_area',
        action='store', type=float, default=60.,
        help='The area of the RoI in m^2')
    parser.add_argument(
        '-roi-length', dest='roi_length',
        action='store', type=float, default=30.,
        help='The length of the RoI in m')
    parser.add_argument(
        '-roi-coordinates', dest='roi_coordinates',
        action='store', type=str, default="0.3,0.3 0.6,0.3 0.6,0.6 0.3,0.6",
        help="""
X,Y Coordinates of RoI in relative values of (0. - 1.)
WARNING: the coordinates must be in the order which adjacent points are connected 
         and the coordinates make a completely closed region
""")

    return parser.parse_args()


if __name__ == '__main__':

    print(time.time())
    args = parse_args()

    run(args)
