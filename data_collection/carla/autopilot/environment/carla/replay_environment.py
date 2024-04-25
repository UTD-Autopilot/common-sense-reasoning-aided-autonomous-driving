import csv
import json
import logging
import os
import pickle
from queue import Queue
import threading

import numpy as np
from PIL import Image
from requests import JSONDecodeError

from ..environment import Environment
from ..msg import ExitMessage, ScheduleCallMessage

class CarlaReplayEnvironment(Environment):
    def __init__(self, record_path):
        super(CarlaReplayEnvironment, self).__init__()
        self.record_path = os.path(record_path)

        with open(os.path.join(self.record_path, 'info.json'), 'r') as info_file:
            info_data = json.load(info_file)

        self.sync = info_data['sync']
        self.tick_interval = info_data['tick_interval']

        self.frame = 0
        self.simulation_time = 0

        self.agent_vehicles = []
        self.tick_callbacks = []

        agent_vehicles_folder = os.path.join(self.record_path, 'agents')
        record_agent_vehicles = []
        for folder in os.listdir(agent_vehicles_folder):
            if not os.path.isdir(os.path.join(agent_vehicles_folder, folder)):
                continue
            if not folder.isnumeric():
                continue
            record_agent_vehicles.append(int(folder))

        record_agent_vehicles.sort()
        if record_agent_vehicles != list(range(len(record_agent_vehicles))):
            logging.warning(f'Agent vehicle save not continuous: {record_agent_vehicles}')

        for vehicle_id in record_agent_vehicles:
            vehicle = CarlaReplyAgentVehicle(os.path.join(self.agent_vehicles_folder, str(vehicle_id)))
            self.agent_vehicles.append(vehicle)
        
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
                self.tick()
        finally:
            pass

    def tick(self):
        self.frame += 1
        for v in self.agent_vehicles:
            v.tick()

        for cb in self.tick_callbacks:
            cb()

    def close(self):
        self.mq.put_nowait(ExitMessage())
        self.thread.join()

class CarlaReplyAgentVehicle:
    def __init__(self, env, record_path):
        self.env = env
        self.record_path = record_path
        if os.path.isfile(os.path.join(self.record_path, 'control.jsonl')):
            self.control = CarlaReplayControl(self.env, os.path.join(self.record_path, 'control.jsonl'))
        else:
            self.control = None
            logging.warning('control.jsonl not found.')

        with open(os.path.join(self.record_path, 'sensors.json'), 'r') as f:
            sensors_info = json.load(f)
        
        self.sensors = {}
        for sensor_name, sensor_info in sensors_info['sensors'].items():
            sensor_record_path = os.path.join(self.record_path, sensor_name)
            if not os.path.exists(sensor_record_path):
                logging.warning(f'Sensor {self.sensor_info["sensor_name"]} record data does not exist.')
            self.sensors[sensor_name] = CarlaReplaySensor(self.env, sensor_record_path, sensor_info)

    def tick(self):
        if self.control is not None:
            self.control.tick()
        for sensor in self.sensors.values():
            sensor.tick()

class CarlaReplayControl:
    def __init__(self, env, record_path):
        self.env = env
        self.record_path = record_path
        self.record = {}

        expeced_keys = ['frame', 'timestamp', 'throttle', 'steer', 'brake', 'hand_brake', 'reverse']
        validated = False
        with open(self.record_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not validated:
                    if not set(expeced_keys).issubset(set(row.keys())):
                        raise RuntimeError('control.csv does not contains all the columns needed.')
                    validated = True
                self.record[row['frame']] = row

    def tick(self):
        frame = self.env.frame
        if frame in self.record:
            self.timestamp = self.record[frame]['timestamp']
            self.throttle = self.record[frame]['throttle']
            self.steer = self.record[frame]['steer']
            self.brake = self.record[frame]['brake']
            self.hand_brake = self.record[frame]['hand_brake']
            self.reverse = self.record[frame]['reverse']

class CarlaReplaySensor:
    def __init__(self, env, record_path, sensor_info):
        self.env = env
        self.record_path = record_path
        self.sensor_info = sensor_info
        self.sensor_type = sensor_info['sensor_type']
        self.listen_callbacks = []
        self.record = {}
        self.data = None
        if self.sensor_type in ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation']:
            # PNG files. We'll read them later.
            pass
        elif self.sensor_type in ['sensor.lidar.ray_cast']:
            # BIN files. We'll read them later.
            pass
        elif self.sensor_type in [
            'sensor.other.imu',
            'sensor.other.gnss',
            'perception.lane.gt'
        ]:
            # CSV files. Preload all into memory.
            with open(os.path.join(self.record_path, 'data.csv'), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.record[row['frame']] = row
        elif self.sensor_type in [
            'perception.traffic_light.gt',
            'perception.vehicle_bbox.gt',
            'perception.pedestrian_bbox.gt',
        ]:
            # Json lines. Preload all into memory.
            with open(os.path.join(self.record_path, 'data.jsonl'), 'r') as f:
                for line in f.readlines():
                    if len(line) == 0:
                        continue
                    data = json.loads(line)
                    self.record[data['frame']] = data
        else:
            # Unknown sensor type, load the pickle later.
            pass

    def listen(self, callback):
        self.listen_callbacks.append(callback)

    def unlisten(self, callback):
        if callback in self.listen_callbacks:
            self.listen_callbacks.remove(callback)

    def tick(self):
        frame = self.env.frame
        if self.sensor_type in ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation']:
            # PNG files.
            image_filepath = os.path.join(self.record_path, str(frame)+'.png')
            if os.path.isfile(image_filepath):
                with Image.open(os.path.join(self.record_path, str(frame)+'.png')) as im:
                    self.data = np.array(im)
        elif self.sensor_type in ['sensor.lidar.ray_cast']:
            # BIN files.
            filepath = os.path.join(self.record_path, str(frame)+'.bin')
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    self.data = f.read()
        elif self.sensor_type in [
            'sensor.other.imu',
            'sensor.other.gnss',
            'perception.lane.gt'
        ]:
            if frame in self.record:
                self.data = self.record[frame]
        elif self.sensor_type in [
            'perception.traffic_light.gt',
            'perception.vehicle_bbox.gt',
            'perception.pedestrian_bbox.gt',
        ]:
            if frame in self.record:
                self.data = self.record[frame]
        else:
            # Python pickle. If it's not we'll simply ignore them.
            filepath = os.path.join(self.record_path, str(frame)+'.pkl')
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    self.data = pickle.load(f)

        for callback in self.listen_callbacks:
            callback(self.fetch())

    def fetch(self):
        return self.data