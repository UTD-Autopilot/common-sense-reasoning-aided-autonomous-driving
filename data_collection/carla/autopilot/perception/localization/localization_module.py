from .state_estimate import StateEstimate
from .rotations import Quaternion
import numpy as np
import math

class LocalizationModule():
    def __init__(self, agent):
        self.agent = agent
        vehicle = agent.vehicle.vehicle
        p = np.array([vehicle.get_transform().location.x, vehicle.get_transform().location.y, vehicle.get_transform().location.z])
        v = np.array([vehicle.get_velocity().x, vehicle.get_velocity().y, vehicle.get_velocity().z])
        q = [math.radians(vehicle.get_transform().rotation.roll), math.radians(vehicle.get_transform().rotation.pitch), math.radians(vehicle.get_transform().rotation.yaw)]
        q = Quaternion(euler=np.array(q)).to_numpy()
        t = 0.0
        self.state = StateEstimate(p, q, v, t)
    
    def tick(self):
        sensors = self.agent.vehicle.sensors
        if 'imu' in sensors:
            if sensors['imu'].fetch() is not None:
                sensor_data = {}
                accelerometer = sensors['imu'].fetch()['accelerometer']
                sensor_data['imu_f'] = np.array(accelerometer)
                gyroscope = sensors['imu'].fetch()['gyroscope']
                sensor_data['imu_w'] = np.array(gyroscope)
                sensor_data['time'] = sensors['imu'].fetch()['timestamp']

                if 'gnss' in sensors:
                    lat = sensors['gnss'].fetch()['latitude']
                    lon = sensors['gnss'].fetch()['longitude']
                    alt = sensors['gnss'].fetch()['altitude']

                    lat_rad = (np.deg2rad(lat) + np.pi) % (2 * np.pi) - np.pi
                    lon_rad = (np.deg2rad(lon) + np.pi) % (2 * np.pi) - np.pi
                    R = 6378135 # Aequator rad
                    x = R * np.sin(lon_rad) * np.cos(lat_rad)
                    y = R * np.sin(-lat_rad)
                    z = alt
                    sensor_data['gnss'] = np.array([x, y, z])

                self.state.updateState(sensor_data)


    def fetch(self):
        # update the object detected.
        # call state.update(dictionary containing sensor data)
        # position is a (x, y, z) vector.
        # velocity is also a (x, y, z) vector.
        # orientation is a quaternion (a, b, c, d).
        # return position, velocity, orientation
        return {
            'position': self.state._p,
            'orientation': self.state._q,
            'velocity': self.state._v,
        }
