from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class RegionOfInterest():
    """
        Create a polygon representing region of interest
        @coordinates: list of coordinates normalized in [0, 100]
        @image_width: width of actual image
        @image_height: height of actual image
        @road_length: actual road distance (m) of the ROI parallel to the direction of traffic
    """
    def __init__(self, coordinates, image_width, image_height, road_area):
        actual_coordinates = []
        for c in coordinates:
            actual_coordinates.append(
                (int(c[0] / 100. * image_width), int(c[1] / 100. * image_height))
            )
        self.roi = Polygon(actual_coordinates)
        if self.roi.area < 1:
            raise Exception("Area of the ROI is 0: please check the coordinates {coordinates}}")
        self.width = image_width
        self.height = image_height
        self.road_area = road_area

    def contains(self, t, b, r, l):
        if r > l:
            x = int((r - l) / 2.)
        else:
            x = int((l - r) / 2.)
        if t > b:
            y = int((t - b) / 2.)
        else:
            y = int((b - t) / 2.)
        p = Point((x, y))
        return self.roi.contains(p)

    def touches(self, t, b, r, l):
        if r > l:
            x = int((r - l) / 2.)
        else:
            x = int((l - r) / 2.)
        if t > b:
            y = int((t - b) / 2.)
        else:
            y = int((b - t) / 2.)
        p = Point((x, y))
        return self.roi.touches(p)

# def load_and_generate_config():
