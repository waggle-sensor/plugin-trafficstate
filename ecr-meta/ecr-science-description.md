# Science
A semi-real-time vehicle tracking application based on deep neural networks was utilized to calculate traffic flow, occupancy and average speed. We utilized YOLO v7 for vehicle detection and SORT method to track the detected vehicles. YOLO v7 is a well-known object detection model based on deep neural networks. SORT is an object tracking method utilizing Kalman filter to calculate possible location of the target between two continuous frames. This application introduced a method that can calculate traffic state independently using a video source.

# AI@Edge
The application first records video for 10 seconds to analyze traffic state. The images are then passed through the YOLO v7 [1] for vehicle detection and SORT [2] for vehicle tracking. The SORT method takes the bounding box of the vehicles and calculates possible location of the vehicle using Kalman filter. With the tracking result, we calculate traffic flow, density (occupancy), and speed.

We set a region of interest (ROI) and line of interest (LOI) is the boundary for all traffic state calculations. We count the number of vehicles that are passing through the LOI for traffic flow calculation. To calculate traffic occupancy, we identify the type of the tacklets (car, truck, and bus) and roungly estimate occupied area over the ROI. For traffic speed calculation, we count the number of frames that each vehicle passes from one end of ROI to the other.

# Using the code
Output: traffic flow, occupancy, and average speed (vehicles/sec, occupied area ratio (0-1), and km/h)  
Input: 10 second video (12 fps, total 120 frames)  
Image resolution (YOLOv7 input resolution): 640x640
Inference time (from object detection to tracking for the whole video):  
Model loading time:  

# Arguments
   '-no-cuda': Do not use CUDA  
   '-stream': ID or name of a stream, e.g. top-camera  
   '-duration': Time duration for input video (default = 10)  
   '-resampling':  Resampling the sample to -resample-fps option (default = 12)  
   '-resampling-fps': Frames per second for input video (default = 12)  
   '-labels': Labels for detection (default = 'detection/coco.names')  
   '-skip-second': Seconds to skip before recording (default = 3)  
   '-roi-name': Name of RoI used when publishing data (default = “incoming”)  
   '-loi-coordinates': X,Y Coordinates of Line of interest for flow calculation (default="0.3,0.3, 0.6,0.3")  
   '-roi-area': The area of the RoI in m^2 (default = 60)  
   '-roi-length': The length of the RoI in m (default = 60)  
   '-roi-coordinates': X,Y Coordinates of RoI in relative values of (0. - 1.) WARNING: the coordinates must be in the order which adjacent points are connected and the coordinates make a completely closed region (default = "0.3,0.3 0.6,0.3 0.6,0.6 0.3,0.6")  
   '-sampling-interval': Inferencing interval for sampling results (default = -1, no samping)  

# Ontology:
The code publishes measurements with optic ‘traffic.state.STATE’, where ‘STATE’ is the state calculated (flow, occupancy, and average_speed).

 
# Inference from Sage codes
To query the output from the plugin, you can do with python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    start="-1h",
    filter={
        "name": "traffic.state.*",
    }
)

# print results in data frame
print(df)
```
For more information, please see [Access and use data documentation](https://docs.waggle-edge.ai/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).

# Reference
[1] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." arXiv preprint arXiv:2207.02696 (2022).
[2] Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft. "Simple online and realtime tracking." In 2016 IEEE international conference on image processing (ICIP), pp. 3464-3468. IEEE, 2016.
