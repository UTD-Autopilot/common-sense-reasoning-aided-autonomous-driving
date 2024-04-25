import random
import threading
import carla
from .sensor import RGBCamera, IMUSensor, GNSSSensor, LiDAR, SemanticSegmentationCamera
from .sensor import GTLocationSensor, GTTrafficLightSensor, GTTrafficSignSensor, GTVehicleBBoxSensor, GTPedestrianBBoxSensor, GTLaneSensor, GTJunctionSensor
from .utils import rpy_to_pyr

class CarlaAgentVehicle:
    def __init__(self, env, vehicle_model='charger_2020', record_mode=False):
        self.env = env
        self.vehicle = None
        self.vehicle_model = vehicle_model
        self.sensors = {}
        self.control = carla.VehicleControl()
        self.spawned = False
        self.autopilot = True
        self.lock = threading.Lock()

    def spawn(self, world, location=None, rotation=None):
        
        bp = world.get_blueprint_library().filter(self.vehicle_model)[0]
        bp.set_attribute('role_name', 'hero')   # Set hero tag

        if location is None:
            self.vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        else:
            self.vehicle = world.spawn_actor(bp, carla.Transform(carla.Location(*location), carla.Rotation(*rpy_to_pyr(rotation))))
        self.vehicle.apply_control(self.control)
        self.vehicle.set_autopilot(self.autopilot, self.env.traffic_manager_port)

        # NuScenes is 1600*900
        camera_image_size_x = 1600
        camera_image_size_y = 900
        camera_fov = 90

        bev_range = 50 # center to edge in meter, so bev size will be double this 
        bev_image_size_x = 500
        bev_image_size_y = 500

        self.sensors['left_front_camera'] = RGBCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=-60)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['front_camera'] = RGBCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=0)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['right_front_camera'] = RGBCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=60)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['left_back_camera'] = RGBCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=-120)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['back_camera'] = RGBCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=180)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['right_back_camera'] = RGBCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=120)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['left_front_semantic_camera'] = SemanticSegmentationCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=-60)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['front_semantic_camera'] = SemanticSegmentationCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=0)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['right_front_semantic_camera'] = SemanticSegmentationCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=60)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['left_back_semantic_camera'] = SemanticSegmentationCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=-120)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['back_semantic_camera'] = SemanticSegmentationCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=180)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['right_back_semantic_camera'] = SemanticSegmentationCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=120)),
            {
                'image_size_x': camera_image_size_x,
                'image_size_y': camera_image_size_y,
                'fov': camera_fov,
            }
        )

        self.sensors['birds_view_camera'] = RGBCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=bev_range), carla.Rotation(pitch=-90)),
            {
                'image_size_x': bev_image_size_x,
                'image_size_y': bev_image_size_y,
                'fov': 90,
            }
        )

        self.sensors['birds_view_semantic_camera'] = SemanticSegmentationCamera(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=bev_range), carla.Rotation(pitch=-90)),
            {
                'image_size_x': bev_image_size_x,
                'image_size_y': bev_image_size_y,
                'fov': 90,
            }
        )

        self.sensors['imu'] = IMUSensor(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=0)),
            {}
        )

        self.sensors['gnss'] = GNSSSensor(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=0)),
            {}
        )

        self.sensors['lidar'] = LiDAR(
            self.vehicle,
            carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(yaw=0)),
            {}
        )

        self.sensors['gt_location'] = GTLocationSensor(self.vehicle, self.env)

        self.sensors['gt_traffic_light'] = GTTrafficLightSensor(
            self.vehicle,
            range=100.0,
        )

        self.sensors['gt_traffic_sign'] = GTTrafficSignSensor(
            self.vehicle,
        )

        self.sensors['gt_vehicle_bbox'] = GTVehicleBBoxSensor(
            self.vehicle, self.env,
        )

        self.sensors['gt_pedestrian_bbox'] = GTPedestrianBBoxSensor(
            self.vehicle,
        )

        self.sensors['gt_lanes'] = GTLaneSensor(
            self.vehicle,
        )

        self.sensors['gt_junctions'] = GTJunctionSensor(
            self.vehicle,
        )

        self.spawned = True

    def tick(self):
        with self.lock:
            if self.vehicle is None:
                return

            for name, sensor in self.sensors.items():
                sensor.tick()

            if not self.autopilot and self.control is not None:
                self.vehicle.apply_control(self.control)

            if self.autopilot:
                self.control = self.vehicle.get_control()

    def destroy(self):
        with self.lock:
            for sensor in self.sensors.values():
                sensor.destroy()
            self.sensors.clear()
            self.vehicle.destroy()
            self.vehicle = None

    def get_velocity(self):
        if not self.vehicle:
            return None

        with self.lock:
            velocity = self.vehicle.get_velocity()
        return velocity
    
    def get_transform(self):
        if not self.vehicle:
            return None
        
        with self.lock:
            transform = self.vehicle.get_transform()
        return transform

    def set_autopilot(self, autopilot, stop_vehicle=True):
        self.autopilot = autopilot
        if self.vehicle is not None:
            self.vehicle.set_autopilot(autopilot, self.env.traffic_manager_port)
            if stop_vehicle:
                self.control.throttle = 0.0
                self.control.brake = 1.0
                self.vehicle.apply_control(self.control)
