import cv2
import numpy as np

from tool.utils import *
from tool.torch_utils import do_detect
from tool.darknet2pytorch import Darknet


def call_yolov4(cfgfile='../yolov4.cfg', weightfile='../yolov4.weights', use_cuda=True):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda().eval()
    else:
        m.eval()

    return m


class yolov4_trck():
    def __init__(self, m):
        self.m = m

    def object_detection(self, frame, ret):
        if ret == False:
            print(ret)
            return 'no_frame'
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sized = cv2.resize(frame, (512, 512))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            #### Start detection using do_detect() function
    #             start = time.time()
            ######### output must be boxes[0], which contains tbrl, confidence level, and class number
            boxes = do_detect(self.m, sized, 0.4, 0.6, True)
    #             print(type(boxes), len(boxes[0]), ': number of detected cars', boxes[0])
    #             finish = time.time()
    #             print('yolo elapsed in: %f sec' % (finish - start))
            return boxes[0]



    def run_yolov4(self, frame, ret):
        return self.object_detection(frame, ret)


if __name__=='__main__':
    video_path='../tracking_record1.mov'
    namesfile='detection/coco.names'

    m = call_yolov4()
    ODetect = yolov4_trck(m)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps:  ', fps)

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    num_classes = m.num_classes
    class_names = load_class_names(namesfile)
#     print(class_names)


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    count = 0
    test = False
    while True:
        count += 1
#         print(count)
        if count == 60*8+1:
            break

        ret, frame = cap.read()

        result = run_yolov4(ODetect, frame, ret)
        print('ex.append(',[result],')')

    cap.release()
