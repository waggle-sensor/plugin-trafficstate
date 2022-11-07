# Traffic state estimator

The plugin calculates traffic flow, occupancy and average speed. It uses the results from vehicle tracking, and it adopts the concept of tracking-by-detection for object tracking. It utilized YOLOv7 or YOLOv7 tiny to detect vehicles, and utilized SORT method to track vehicles. The plugin requires Nvidia GPU for inferencing. To calculate traffic state on the region where interested in (for example, 4 right lanes where the vehicles are comming in) region of interest must be provided to the plugin. ROI need to be provided as the points of the corner of the ploygon shaped the ROI. The points are the points in ratios of the image (If one point is the center of the image, then the point need to be written as 0.5,0.5).

## How to use

```bash
# -stream option is required to indicate source of a streaming input
# for example, python3 app.py -stream bottom
#   images from the camera named "bottom"

# to designate region of interest and line of interest
$ python3 app.py -stream bottom -roi-area 75 -roi-length 30 -roi-coordinates "0.47265625,1.0 0.6796875,0.859375 0.30078125,0.15625 0.21875,0.16667 0.2265625,0.23958 0.27734375,0.46354 0.3359375,0.671875" -loi-coordinates "0.30078125,0.15625 0.21875,0.16667"
```

## Notes to users:
The code requires some of input parameters (you can see the parameters in the app.py):
-	-stream: path of the input video
-	-loi-coordinates: coordinate of line of interest in ratio in the image
-	-roi-area: area in square meter of region of interest
-	-roi-length: length in meter of region of interest
-	-roi-coordinates: coordinate of region of interest in ratio in the image

The loi-coordinate and roi-coordinate are the positions pointing our region of interest in the ratio. So for example if we want to see the vehicles that drive through a rectangle shaped from (0,0) to (256,256) in 512x512 image, then the input must be (0,0 0.5,0 0.5,0.5 0,0.5). It is the same with the line of interest.

## funding
[NSF 1935984](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1935984)

## collaborators
Bhupendra Raut, Dario Dematties Reyes, Joseph Swantek, Neal Conrad, Nicola Ferrier, Pete Beckman, Raj Sankaran, Robert Jackson, Scott Collis, Sean Shahkarami, Seongha Park, Sergey Shemyakin, Wolfgang Gerlach, Yongho kim
