import os
import threading
import time
from queue import Queue

import numpy as np
from nuscenes import NuScenes
from PIL import Image

from ..environment import Environment, DummyControl
from ..msg import ExitMessage, ScheduleCallMessage

class NuScenesEnvironment(Environment):
    def __init__(
        self,
        version='v1.0-mini',
        dataroot=None,
    ):
        self.nusc_version = version
        self.nusc_dataroot = dataroot
        self.nusc = NuScenes(version=version, dataroot=dataroot)

        self.agent_vehicles = []
        self.tick_callbacks = []

        self.current_scene = self.nusc.scene[0]
        self.current_sample_token = self.current_scene['first_sample_token']
        self.current_sample_number = 0
        self.current_sample = None
        self.timestamp = None

        # Load the first sample
        self.tick()

        self.paused = False

        self.mq = Queue()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        try:
            exit = False
            while True:
                while not self.mq.empty():
                    msg = self.mq.get_nowait()
                    if isinstance(msg, ExitMessage):
                        exit = True
                    elif isinstance(msg, ScheduleCallMessage):
                        msg.func(*msg.args, **msg.kwargs)
                if exit:
                    break

                if self.paused:
                    time.sleep(0.01)
                    continue

                self.tick()
        finally:
            pass
    
    def tick(self):
        if self.current_sample is not None:
            if len(self.current_sample['next']) == 0:
                self.paused = True
                return
            else:
                self.current_sample_token = self.current_sample['next']
                self.current_sample_number += 1

        if self.current_sample_token is not None:
            try:
                self.current_sample = self.nusc.get('sample', self.current_sample_token)
                self.timestamp = self.current_sample['timestamp']
            except Exception as e:
                print('Failed to get sample from NuScenes.')
                print(e)

            for v in self.agent_vehicles:
                v.tick()
            for cb in self.tick_callbacks:
                cb()
    
    def spawn_agent_vehicle(self):
        vehicle = NuScenesAgentVehicle(self)
        self.agent_vehicles.append(vehicle)
        return vehicle

    def despawn_agent_vehicle(self, vehicle):
        if vehicle in self.agent_vehicles:
            self.agent_vehicles.remove(vehicle)
            vehicle.destroy()

    def despawn_all_agents(self):
        for vehicle in self.agent_vehicles:
            self.agent_vehicles.remove(vehicle)
            vehicle.destroy()

    def add_tick_callback(self, tick_callback):
        self.tick_callbacks.append(tick_callback)

    def remove_tick_callback(self, tick_callback):
        if tick_callback in self.tick_callbacks:
            self.tick_callbacks.remove(tick_callback)

    def close(self):
        self.mq.put_nowait(ExitMessage())
        self.thread.join()
    
    def get_scenes(self):
        return self.nusc.scene
    
    def change_scene(self, scene_idx):
        def change_scene_call(scene_idx):
            self.current_scene = self.nusc.scene[scene_idx]
            self.current_sample_token = self.current_scene['first_sample_token']
            self.current_sample_number = 0
            self.current_sample = None
            self.paused = False
        self.mq.put_nowait(ScheduleCallMessage(change_scene_call, scene_idx))

    def seek_by_sample_number(self, sample_number):
        def seek_by_sample_number_call(sample_number):
            self.current_sample_token = self.current_scene['first_sample_token']
            self.current_sample_number = 0
            self.current_sample = self.nusc.get('sample', self.current_sample_token)
            while self.current_sample_number < sample_number:
                if len(self.current_sample['next']) == 0:
                    self.paused = True
                    return
                else:
                    self.current_sample_token = self.current_sample['next']
                    self.current_sample_number += 1
                    self.current_sample = self.nusc.get('sample', self.current_sample_token)
        self.mq.put_nowait(ScheduleCallMessage(seek_by_sample_number_call, sample_number))


class NuScenesAgentVehicle():
    def __init__(self, env):
        self.env = env

        self.autopilot = True
        self.control = DummyControl()
        self.sensors = {}

        self.sensors['left_front_camera'] = NuScenesCamera(self.env, 'CAM_FRONT_LEFT')
        self.sensors['front_camera'] = NuScenesCamera(self.env, 'CAM_FRONT')
        self.sensors['right_front_camera'] = NuScenesCamera(self.env, 'CAM_FRONT_RIGHT')
        self.sensors['left_back_camera'] = NuScenesCamera(self.env, 'CAM_BACK_LEFT')
        self.sensors['back_camera'] = NuScenesCamera(self.env, 'CAM_BACK')
        self.sensors['right_back_camera'] = NuScenesCamera(self.env, 'CAM_BACK_RIGHT')
        self.sensors['lidar'] = NuScenesLidar(self.env, 'LIDAR_TOP')

        # TODO: IMU, NGSS, GT Sensors
    
    def set_autopilot(self, autopilot):
        pass
    
    def tick(self):
        for sensor_name, sensor in self.sensors.items():
            sensor.tick()

class NuSceneSensor:
    def __init__(self, env, sensor_name):
        self.env = env
        self.sensor_name = sensor_name

        self.listen_callbacks = []

        self.data_token = None
        self.raw_data = None

    def listen(self, callback):
        self.listen_callbacks.append(callback)

    def unlisten(self, callback):
        if callback in self.listen_callbacks:
            self.listen_callbacks.remove(callback)

    def tick(self):
        self.data_token = self.env.current_sample['data'][self.sensor_name]
        self.sample_data = self.env.nusc.get('sample_data', self.data_token)
        self.data_path = os.path.join(self.env.nusc_dataroot, self.sample_data['filename'])
        fileformat = self.sample_data['fileformat']

        if fileformat == 'jpg' or fileformat == 'png':
            # Load JPG image
            with Image.open(self.data_path) as im:
                self.data = np.array(im)
        elif fileformat == 'pcd':
            # Load PCD point cloud
            scan = np.fromfile(self.data_path, dtype=np.float32)
            # x, y, z, intensity, ring_index
            # We don't need the ring index now so we'll drop that to match the CARLA lidar format.
            self.data = scan.reshape((-1, 5))[:, :4].copy()
        else:
            raise RuntimeError(f'Unsupported file format {fileformat}')

        for callback in self.listen_callbacks:
            callback(self.fetch())

    def fetch(self):
        return self.on_fetch()

    def on_fetch(self):
        return self.data

    def destroy(self):
        pass

class NuScenesCamera(NuSceneSensor):
    def __init__(self, env, sensor_name):
        super(NuScenesCamera, self).__init__(env, sensor_name)
        self.sensor_type = 'sensor.camera.rgb'
        self.sensor_options = {
            'image_size_x': 1600,
            'image_size_y': 900,
        }
    def on_fetch(self):
        return self.data

class LidarMeasurement:
    def __init__(self, raw_data):
        self.raw_data = raw_data

class NuScenesLidar(NuSceneSensor):
    def __init__(self, env, sensor_name):
        super(NuScenesLidar, self).__init__(env, sensor_name)
        self.sensor_type = 'sensor.lidar.ray_cast'
        self.lidar_range = 150
    
    def on_fetch(self):
        return LidarMeasurement(self.data)

    def lidar_image(self, data, image_size=(400, 400)):
        self.lidar_range = 150
        lidar_range = 2.0 * float(self.lidar_range)

        points = data.raw_data
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

class NuScenesRadar(NuSceneSensor):
    def __init__(self, env, sensor_name):
        super(NuScenesRadar, self).__init__(env, sensor_name)
        self.sensor_type = 'sensor.radar.ray_cast'
