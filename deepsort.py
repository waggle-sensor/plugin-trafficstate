from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.utils import preprocessing as prep

from scipy.stats import multivariate_normal

import torch
import torchvision

import numpy as np

def get_gaussian_mask():
    #128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma**2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask


class deepsort_rbc():
    def __init__(self, m_deepsort):
        self.encoder = m_deepsort
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', .5, 100)  ## euclidian or cosine
        self.tracker = Tracker(self.metric)

        self.gaussian_mask = get_gaussian_mask().cuda()

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])


    def pre_process(self, frame, boxes):
        crops = []

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])

        width = frame.shape[1]
        height = frame.shape[0]
#         print(width, height)
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            try:
                crop = frame[y1:y2, x1:x2, :]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)
        return crops


#     def run_deep_sort(self, frame, out_scores, out_boxes):
    def a_run_deep_sort(self, frame, boxes, width, height):

#         if out_boxes == []:
        if boxes == None or boxes == []:
            self.tracker.predict()
            print('no detections')
            trackers = self.tracker.tracks
            return trackers

        processed_crops = self.pre_process(frame, boxes).cuda()
        processed_crops = self.gaussian_mask * processed_crops

        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()

#         print('features', features)

        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        detections = []
        out_scores = []
        for i in range(len(boxes)):
            detect = boxes[i][:4]
            detect[0] = int(detect[0] * width)
            detect[1] = int(detect[1] * height)

            detect[2] = int(detect[2] * width)
            detect[3] = int(detect[3] * height)
            detect[2] = detect[2] - detect[0]
            detect[3] = detect[3] - detect[1]

            detections.append(detect)
            out_scores.append(boxes[i][5])
        detections = np.asarray(detections)
        out_scores = np.asarray(out_scores)

        dets = [Detection(bbox, score, feature) for bbox, score, feature in zip(detections, out_scores, features)]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker, dets, features
