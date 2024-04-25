import math
import traceback
import copy

import carla
import numpy as np
from .cord import create_bb_points, actor_to_world, world_to_sensor
from transforms3d.euler import euler2mat
from .utils import carla_bbox_to_dict
from scipy.spatial import distance

class Sensor:
    def __init__(self, parent, sensor_type):
        self.parent = parent
        self.sensor = None
        self.sensor_type = sensor_type
        self.world = self.parent.get_world()

        self.listen_callbacks = []

    def listen(self, callback):
        self.listen_callbacks.append(callback)

    def unlisten(self, callback):
        if callback in self.listen_callbacks:
            self.listen_callbacks.remove(callback)

    def tick(self):
        for callback in self.listen_callbacks:
            callback(self.fetch())

    def fetch(self):
        return self.on_fetch()

    def on_fetch(self):
        raise NotImplementedError()

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
        self.listen_callbacks = []


class GeneralSensor(Sensor):
    # General sensor wrapper
    def __init__(self, sensor_type, parent, transform, sensor_options):
        super().__init__(parent, sensor_type)
        self.sensor = self.init_sensor(self.sensor_type, transform, parent, sensor_options)
        self.transform = transform
        self.sensor_options = sensor_options
        self.data = None

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        bp = self.world.get_blueprint_library().find(sensor_type)
        for key, value in sensor_options.items():
            bp.set_attribute(key, str(value))

        sensor = self.world.spawn_actor(bp, transform, attach_to=attached)
        sensor.listen(self.save_data)
        return sensor

    def save_data(self, data):
        self.data = data

    def on_fetch(self):
        return self.data


class RGBCamera(Sensor):
    def __init__(self, parent, transform, sensor_options):
        super().__init__(parent, 'sensor.camera.rgb')
        self.transform = transform
        self.camera_bp = None
        self.sensor = self.init_sensor(self.sensor_type, transform, parent, sensor_options)
        self.sensor_options = sensor_options
        self.data = None
        self.tics_processing = 0

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        self.camera_bp = self.world.get_blueprint_library().find(sensor_type)

        for key, value in sensor_options.items():
            self.camera_bp.set_attribute(key, str(value))

        camera = self.world.spawn_actor(self.camera_bp, transform, attach_to=attached)

        calibration = np.identity(3)
        calibration[0, 2] = sensor_options['image_size_x'] / 2.0
        calibration[1, 2] = sensor_options['image_size_y'] / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_options['image_size_x'] / (
                    2.0 * np.tan(sensor_options['fov'] * np.pi / 360.0))
        camera.calibration = calibration
        camera.listen(self.save_rgb_image)

        return camera

    def get_camera_info(self):
        translation = [self.transform.location.x, self.transform.location.y, self.transform.location.z]

        roll = math.radians(self.transform.rotation.roll-90)
        pitch = -math.radians(self.transform.rotation.pitch)
        yaw = -math.radians(self.transform.rotation.yaw)
        rotation_matrix = euler2mat(roll, pitch, yaw)
        intrinsics = self.sensor.calibration

        return rotation_matrix, translation, intrinsics

    def save_rgb_image(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        img = array.copy()

        self.data = img
        self.tics_processing += 1

    def on_fetch(self):
        return self.data

    def get_object_bounding_box(self, object):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        bb_cords = create_bb_points(object)
        world_cord = actor_to_world(bb_cords, object)
        cords_x_y_z = world_to_sensor(world_cord, self.sensor)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(self.sensor.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)

        return camera_bbox


class SemanticSegmentationCamera(Sensor):
    def __init__(self, parent, transform, sensor_options):
        super().__init__(parent, 'sensor.camera.semantic_segmentation')
        self.transform = transform
        self.sensor = self.init_sensor(self.sensor_type, transform, parent, sensor_options)
        self.sensor_options = sensor_options
        self.data = None
        self.tics_processing = 0

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        camera_bp = self.world.get_blueprint_library().find(sensor_type)

        for key, value in sensor_options.items():
            camera_bp.set_attribute(key, str(value))

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)

        calibration = np.identity(3)
        calibration[0, 2] = sensor_options['image_size_x'] / 2.0
        calibration[1, 2] = sensor_options['image_size_y'] / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_options['image_size_x'] / (
                    2.0 * np.tan(sensor_options['fov'] * np.pi / 360.0))
        camera.calibration = calibration

        camera.listen(self.save_rgb_image)
        return camera

    def get_object_bounding_box(self, object):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        bb_cords = create_bb_points(object)
        world_cord = actor_to_world(bb_cords, object)
        cords_x_y_z = world_to_sensor(world_cord, self.sensor)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(self.sensor.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)

        return camera_bbox

    def save_rgb_image(self, image):
        try:
            image.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            img = array.copy()

            self.data = img
            self.tics_processing += 1

        except Exception as e:
            print(e)

    def on_fetch(self):
        return self.data

    def visualize_rgb(self, img):
        return img

    def map_color(self, img, target, old, new):
        mask = np.all(img == old, axis=-1)
        target[mask, :] = new
        return target


class LidarMeasurement:
    def __init__(self, channels, horizontal_angle, raw_data):
        self.channels = channels
        self.horizontal_angle = horizontal_angle
        self.raw_data = bytes(raw_data)


class LiDAR(Sensor):
    def __init__(self, parent, transform, sensor_options):
        super().__init__(parent, 'sensor.lidar.ray_cast')
        self.transform = transform
        self.sensor_options = None
        self.sensor = self.init_sensor(self.sensor_type, transform, parent, sensor_options)
        self.data = None
        self.tics_processing = 0

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        bp = self.world.get_blueprint_library().find(sensor_type)

        # Velodyne HDL-64E assuming 20 FPS
        options = {
            'range': 80,
            'points_per_second': 2600000,
            'rotation_frequency': 40,
            'channels': 64,
            'upper_fov': 2,
            'lower_fov': -24.8
        }

        options.update(sensor_options)

        self.sensor_options = options

        for key, value in self.sensor_options.items():
            bp.set_attribute(key, str(value))

        lidar = self.world.spawn_actor(bp, transform, attach_to=attached)
        lidar.listen(self.save_data)
        return lidar

    def save_data(self, data):
        self.data = data
        # self.data = LidarMeasurement(data.channels, data.horizontal_angle, data.raw_data)
        self.tics_processing += 1

    def lidar_image(self, data, image_size=(500, 500)):
        lidar_range = 2.0 * float(self.sensor_options['range'])

        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(image_size) / lidar_range
        lidar_data += (0.5 * image_size[0], 0.5 * image_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (image_size[0], image_size[1], 3)
        lidar_img = np.zeros(lidar_img_size, dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        return lidar_img

    def on_fetch(self):
        return self.data


class IMUSensor(GeneralSensor):
    def __init__(self, parent, transform, sensor_options):
        super().__init__('sensor.other.imu', parent, transform, sensor_options)
        self.data = None

    def save_data(self, data):
        self.data = {
            'frame': data.frame,
            'timestamp': data.timestamp,
            'accelerometer': [data.accelerometer.x, data.accelerometer.y, data.accelerometer.z],
            'compass': data.compass,
            'gyroscope': [data.gyroscope.x, data.gyroscope.y, data.gyroscope.z],
        }

    def on_fetch(self):
        return self.data


class GNSSSensor(GeneralSensor):
    def __init__(self, parent, transform, sensor_options):
        super().__init__('sensor.other.gnss', parent, transform, sensor_options)
        self.data = None

    def save_data(self, data):
        self.data = {
            'frame': data.frame,
            'timestamp': data.timestamp,
            'latitude': data.latitude,
            'longitude': data.longitude,
            'altitude': data.altitude,
        }

    def on_fetch(self):
        return self.data


class CollisionSensor(GeneralSensor):
    def __init__(self, parent, transform, sensor_options):
        super().__init__('sensor.other.collision', parent, transform, sensor_options)


class GroundTruthSensor(Sensor):
    pass


class GTLocationSensor(GroundTruthSensor):
    def __init__(self, parent, env):
        super().__init__(parent, 'perception.location.gt')
        self.env = env
    
    def on_fetch(self):
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame

        transform = self.parent.get_transform()
        location = [transform.location.x, transform.location.y, transform.location.z]
        rotation = [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw]

        # Determine the high-level action of the vehicle by its light state
        try:
            action = self.env.traffic_manager.get_next_action(self.parent)[0]
        except Exception as e:
            action = "Unknown"
            #traceback.print_exc()

        return {
            'frame': frame,
            'timestamp': timestamp,
            'location': location,
            'rotation': rotation,
            'current_action': action,
        }

def traffic_light_state_to_string(state):
    if state == carla.TrafficLightState.Red:
        return 'Red'
    elif state == carla.TrafficLightState.Yellow:
        return 'Yellow'
    elif state == carla.TrafficLightState.Green:
        return 'Green'
    elif state == carla.TrafficLightState.Off:
        return 'Off'
    elif state == carla.TrafficLightState.Unknown:
        return 'Unknown'
    else:
        return 'ERROR'

def get_first_vehicle_in_a_row(vehicle, min_distance=15.0):
    while True:
        next_vehicle = None
        cur_distance = min_distance
        transform = vehicle.get_transform()
        forward_vector = transform.get_forward_vector()
        vehicles = vehicle.get_world().get_actors().filter('vehicle.*')
        
        for v2 in vehicles:
            if v2.id == vehicle.id:
                continue
            v2_distance = vehicle.get_location().distance(v2.get_location())
            if v2_distance > min_distance:
                continue
            vec2 = v2.get_location() - vehicle.get_location()
            if (forward_vector.get_vector_angle(vec2) < 0.33*np.pi
                and forward_vector.get_vector_angle(v2.get_transform().get_forward_vector()) < 0.33*np.pi):
                if v2_distance <= cur_distance or next_vehicle is None:
                    cur_distance = v2_distance
                    next_vehicle = v2

        if next_vehicle is not None:
            vehicle = next_vehicle
            if vehicle.is_at_traffic_light():
                break
        else:
            break

    return vehicle

class GTTrafficLightSensor(GroundTruthSensor):
    def __init__(self, parent, range=1e6):
        super().__init__(parent, 'perception.traffic_light.gt')
        self.range = range

    def on_fetch(self):
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame
        traffic_lights = []
        #pl = self.parent.get_transform().location
        for tl in self.world.get_actors().filter('traffic.traffic_light*'):
            if tl.get_transform().location.distance(self.parent.get_transform().location) < self.range:
                #verts = [[v.x - pl.x, v.y - pl.y, v.z - pl.z] for v in bbox.get_world_vertices(carla.Transform())]
                bbox = tl.bounding_box
                location = tl.get_transform().location
                rotation = tl.get_transform().rotation

                light_state = traffic_light_state_to_string(tl.get_state())

                traffic_lights.append({
                    'id': str(tl.id),
                    'location': [location.x, location.y, location.z],
                    'rotation': [rotation.roll, rotation.pitch, rotation.yaw],
                    'bbox': carla_bbox_to_dict(bbox),
                    'state': light_state,
                })
        # get the traffic light affecting us
        is_at_traffic_light = False
        current_traffic_light = None

        first_vehicle = get_first_vehicle_in_a_row(self.parent)

        if first_vehicle.is_at_traffic_light():
            is_at_traffic_light = True
            tl = first_vehicle.get_traffic_light()
            bbox = tl.bounding_box
            location = tl.get_transform().location
            rotation = tl.get_transform().rotation
            light_state = traffic_light_state_to_string(tl.get_state())
            current_traffic_light = {
                'id': str(tl.id),
                'location': [location.x, location.y, location.z],
                'rotation': [rotation.roll, rotation.pitch, rotation.yaw],
                'bbox': carla_bbox_to_dict(bbox),
                'state': light_state,
                'first_vehicle_id': first_vehicle.id,
            }

        return {
            'frame': frame,
            'timestamp': timestamp,
            'traffic_lights': traffic_lights,
            'is_at_traffic_light': is_at_traffic_light,
            'current_traffic_light': current_traffic_light,
        }

class GTTrafficSignSensor(GroundTruthSensor):
    def __init__(self, parent, range=1e6):
        super().__init__(parent, 'perception.traffic_sign.gt')
        self.range = range

    def on_fetch(self):
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame
        objects = self.world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
        traffic_signs = []
        #pl = self.parent.get_transform().location
        for obj in self.world.get_actors().filter('traffic.traffic_sign*'):
            
            if obj.get_transform().location.distance(self.parent.get_transform().location) < self.range:
                #verts = [[v.x - pl.x, v.y - pl.y, v.z - pl.z] for v in bbox.get_world_vertices(carla.Transform())]
                bbox = obj.bounding_box
                location = obj.get_transform().location
                rotation = obj.get_transform().rotation
                traffic_signs.append({
                    'id': str(obj.id),
                    'location': [location.x, location.y, location.z],
                    'rotation': [rotation.roll, rotation.pitch, rotation.yaw],
                    'bbox': carla_bbox_to_dict(bbox),
                })
        return {
            'frame': frame,
            'timestamp': timestamp,
            'traffic_signs': traffic_signs,
        }


class GTVehicleBBoxSensor(GroundTruthSensor):
    def __init__(self, parent, env, range=1e6):
        super().__init__(parent, 'perception.vehicle_bbox.gt')
        # Filter for the vehicles within max_distance
        self.range = range
        self.env = env

    def on_fetch(self):
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame
        vehicles = []
        pl = self.parent.get_transform().location
        for vehicle in self.world.get_actors().filter('*vehicle*'):
            if vehicle.id == self.parent.id:
                continue
            bbox = vehicle.bounding_box
            dist = vehicle.get_transform().location.distance(self.parent.get_transform().location)
            
            if dist < self.range:
                #verts = [[v.x - pl.x, v.y - pl.y, v.z - pl.z] for v in bbox.get_world_vertices(carla.Transform())]
            
                # Determine the high-level action of the vehicle by its light state
                try:
                    action = self.env.traffic_manager.get_next_action(vehicle)[0]
                except Exception as e:
                    action = "Unknown"
                    #traceback.print_exc()

                location = vehicle.get_transform().location
                rotation = vehicle.get_transform().rotation
                velocity = vehicle.get_velocity()
                vehicles.append({
                    'id': str(vehicle.id),
                    'current_action': action,
                    'location': [location.x, location.y, location.z],
                    'rotation': [rotation.roll, rotation.pitch, rotation.yaw],
                    'velocity': [velocity.x, velocity.y, velocity.z],
                    'bbox': carla_bbox_to_dict(bbox),
                })

        return {
            'frame': frame,
            'timestamp': timestamp,
            'vehicles': vehicles,
        }


class GTPedestrianBBoxSensor(GroundTruthSensor):
    def __init__(self, parent, range=1e6):
        super().__init__(parent, 'perception.pedestrian_bbox.gt')
        self.range = range

    def on_fetch(self):
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame
        pedestrians = []
        pl = self.parent.get_transform().location
        for npc in self.world.get_actors().filter('*pedestrian*'):
            if npc.id != self.parent.id:
                bbox = npc.bounding_box
                dist = npc.get_transform().location.distance(self.parent.get_transform().location)
                
                if dist < self.range:
                    #verts = [[v.x - pl.x, v.y - pl.y, v.z - pl.z] for v in bbox.get_world_vertices(carla.Transform())]
                    location = npc.get_transform().location
                    rotation = npc.get_transform().rotation
                    velocity = npc.get_velocity()
                    pedestrians.append({
                        'id': str(npc.id),
                        'location': [location.x, location.y, location.z],
                        'rotation': [rotation.roll, rotation.pitch, rotation.yaw],
                        'velocity': [velocity.x, velocity.y, velocity.z],
                        'bbox': carla_bbox_to_dict(bbox)
                    })
        return {
            'frame': frame,
            'timestamp': timestamp,
            'pedestrians': pedestrians,
        }


class GTLaneSensor(GroundTruthSensor):
    def __init__(self, parent):
        super().__init__(parent, 'perception.lane.gt')

    def on_fetch(self):
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame

        map = self.world.get_map()
        waypoint = map.get_waypoint(self.parent.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            return {
                'frame': frame,
                'timestamp': timestamp,
                'has_lane': False,
                'left_lanes': 0,
                'right_lanes': 0,
            }
        # print('w', waypoint)
        num_left_lane = 0
        left_lane = waypoint.get_left_lane()
        if left_lane is not None:
            # print('l', left_lane)
            if left_lane != waypoint:
                num_left_lane += 1
        num_right_lane = 0
        right_lane = waypoint.get_right_lane()
        if right_lane is not None:
            # print('r', right_lane)
            if right_lane != waypoint:
                num_right_lane += 1

        return {
            'frame': frame,
            'timestamp': timestamp,
            'has_lane': True,
            'left_lanes': num_left_lane,
            'right_lanes': num_right_lane,
        }

class GTJunctionSensor(GroundTruthSensor):
    def __init__(self, parent, range=1e6):
        super().__init__(parent, 'perception.junction.gt')
        self.range = range

    def on_fetch(self):
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        frame = snapshot.frame

        map = self.world.get_map()
        topology = map.get_topology()
        waypoints = [w for p in topology for w in p]

        vehicle_location = None
        if self.parent is not None:
            vehicle_location = self.parent.get_transform().location
            vehicle_location = [vehicle_location.x, vehicle_location.y, vehicle_location.z]

        junctions = {} # id, junction
        for wp in waypoints:
            if wp.is_junction:
                junction = wp.get_junction()
                location = junction.bounding_box.location
                rotation = junction.bounding_box.rotation

                bbox = carla_bbox_to_dict(junction.bounding_box)

                if vehicle_location is not None:
                    d = distance.euclidean(vehicle_location, [location.x, location.y, location.z])
                    if d > distance.euclidean([0, 0, 0], bbox['extent']) + self.range:
                        continue
                
                junctions[str(junction.id)] = {
                    'id': str(junction.id),
                    'location': [location.x, location.y, location.z],
                    'rotation': [rotation.roll, rotation.pitch, rotation.yaw],
                    'bounding_box': bbox
                }
        
        return {
            'frame': frame,
            'timestamp': timestamp,
            'junctions': list(junctions.values()),
        }
