# Science
A semi-real-time vehicle tracking application based on deep neural networks was utilized to calculate traffic state. We utilized machine learning methods to detect and track the traffic -- YOLO v4 for traffic detection and DeepSORT method to track the detected vehicles. YOLO v4 is a well-known object detection model based on deep neural networks. DeepSORT is an object tracking method utilizing Siamese network to calculate similarity between the same target in two continuous frames and Kalman filter, Cosine distance, and Mahalanobis distance to calculate possible location of the target between two continuous frames, and Hungarian method to associate of detection and tracklet. and The traffic state has been calculated by ‘traffic flow = traffic speed x traffic density’ [1-2]. With the equation, we need to measure two of the parameters using instruments for each parameter to estimate the third state. However, this application introduced based on deep neural networks can calculate traffic state independently using a video source.

We tried to estimate traffic states directly from the information that can be extracted from videos. The traffic state, that is traffic volume, space occupancy, travel speed, travel time and travel delays can be categorized into three types of information: traffic speed, flow and density [1]. We calculated the three traffic states, flow, speed and density, which can be represented as traffic volume, travel speed or time and space occupancy respectively. The image processing algorithm required videos that were recorded with frame rate higher than 12 fps. The calculation result reporting frequency can be determined based on the demand of the user.

 
# AI@Edge
The application first records video for 30 seconds to analyze traffic state. Because it takes about 1 second to analyze a frame of image but it requires more than 12 fps to track vehicles, we record video first and then analyze that. The images are then passed through the YOLO v4 [3] for vehicle detection and DeepSORT [4] for vehicle tracking. The DeepSORT method takes the bounding box of the vehicles and calculates similarity of the detection and tracklet in two continuous frames using Siamese network [5] and calculates possible location of the vehicle using Kalman filter. After that the method calculates distance between detection and tracklet using Cosine distance and Mahalanobis distance, and associates detection and tracklet using Hungarian method. Through the series of processes, the DeepSORT method tracks vehicles. With the tracking ability, we calculate traffic flow, density (occupancy), and speed.

We set a region of interest (ROI), and the ROI is the boundary for all traffic state calculations. We count the number of vehicles that are passing through the ROI for traffic flow calculation. To calculate traffic occupancy, we identify the type of the tacklets (car, truck, and bus) and measure occupied area over the ROI. For traffic speed calculation, we count the number of frames that each vehicle passes from one end of ROI to the other.

# Using the code
Output: traffic flow, occupancy, and speed (vehicles/sec, occupied area ratio (0-1), and km/h)  
Input: 10 second video (12 fps, total 120 frames)  
Image resolution (YOLOv4 input resolution): 512x512  
Inference time (from object detection to tracking for the whole video):  
Model loading time (for both YOLO v4 and DeepSORT):  

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
The code publishes measurements with optic ‘env.traffic.STATE’, where ‘STATE’ is the state calculated (flow, occupancy, and speed).

 
# References
[1] Fred L. Hall, "Traffic stream characteristics," Traffic Flow Theory, US Federal Highway Administration, 36, 1996.  
[2] Leslie C. Edie “Discussion of traffic stream measurements and definitions,” New York: Port of New York Authority, 1963.  
[3] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object detection." arXiv preprint arXiv:2004.10934, 2020.  
[4] Nicolai Wojke, and Alex Bewley. "Deep cosine metric learning for person re-identification." In 2018 IEEE winter conference on applications of computer vision (WACV), pp. 748-756. IEEE, 2018.  
[5] Gregory ​​Koch, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." In ICML deep learning workshop, vol. 2. 2015.
