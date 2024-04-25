class ObjectDetectionModule():
    def __init__(self, vehicle):
        self.vehicle = vehicle
    
    def tick(self):
        pass
    
    def fetch(self):
        # update the object detected.
        # return [
        #     {
        #         "type": "vehicle",
        #         "bounding_box": [x1, y1, z1, x2, y2, z2],
        #     },
        #     {
        #         "type": "traffic_light",
        #         "bounding_box": [x1, y1, z1, x2, y2, z2],
        #     },
        # ]
        return NotImplementedError()
    