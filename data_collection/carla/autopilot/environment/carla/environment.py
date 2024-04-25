import carla
import datetime
import json
import logging
import os
import threading
import time
import numpy as np
import pickle
from PIL import Image
from queue import Queue
import random
import traceback
import gc
import copy

from . import RGBCamera, SemanticSegmentationCamera, LiDAR
from ..environment import Environment, Recorder
from ..msg import ExitMessage, ScheduleCallMessage
from .vehicle import CarlaAgentVehicle
from .utils import find_free_port

class CarlaRuntimeError(RuntimeError):
    def __init__(self, error) -> None:
        self.error = error
        super.__init__(self)

class CarlaEnvironment(Environment):
    def __init__(
            self,
            carla_host='127.0.0.1',
            carla_port='2000',
            carla_timeout=30.0,
            sync=True,
            tick_interval=0.5, # Must be multiple a of 0.05
    ):
        super(CarlaEnvironment, self).__init__()

        self.carla_host = carla_host
        self.carla_port = carla_port
        self.carla_timeout = carla_timeout
        self.sync = sync
        self.tick_interval = tick_interval

        self.fixed_delta_seconds = 0.05
        self.skip_frame_number = int(tick_interval / self.fixed_delta_seconds)

        if self.skip_frame_number != tick_interval / self.fixed_delta_seconds:
            raise RuntimeError("tick_interval must be a multiple of 0.05")

        self.frame = 0
        self.timestamp = 0.0

        self.agent_vehicles = []
        self.tick_callbacks = []

        self.vehicles_list = []
        self.walkers_list = []
        self.obstacle_list = []
        self.all_id = []
        self.all_actors = []

        self.scenario_ego_location = None
        self.scenario_ego_rotation = None

        # Spectator following object for exploring and debugging.
        self.spectator_following_object = None

        self.carla = carla.Client(self.carla_host, self.carla_port)
        self.carla.set_timeout(self.carla_timeout)
        self.traffic_manager_port = find_free_port()
        self.traffic_manager = self.carla.get_trafficmanager(self.traffic_manager_port)

        self.world = self.carla.get_world()
        
        self.apply_world_settings()

        self.mq = Queue()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        exit = False
        while True:
            while not self.mq.empty():
                msg = self.mq.get_nowait()
                if isinstance(msg, ExitMessage):
                    exit = True
                elif isinstance(msg, ScheduleCallMessage):
                    try:
                        msg.func(*msg.args, **msg.kwargs)
                    except Exception as e:
                        traceback.print_exc()
            if exit:
                break

            try:
                self.tick()
            except CarlaRuntimeError as e:
                traceback.print_exc()
                return
            except Exception as e:
                traceback.print_exc()

        self.despawn_all_agents()
        self.despawn_traffic()

        if self.carla is not None:
            self.carla.apply_batch([carla.command.DestroyActor(x.vehicle) for x in self.agent_vehicles])
            self.world.apply_settings(self.original_settings)
        self.agent_vehicles = []
        self.tick_callbacks = []

    def tick(self):
        snapshot = self.world.get_snapshot()
        self.timestamp = snapshot.timestamp.elapsed_seconds
        self.frame = snapshot.frame
        for v in self.agent_vehicles:
            v.tick()

        for cb in self.tick_callbacks:
            cb()

        for _ in range(self.skip_frame_number):
            try:
                if self.sync:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
            except RuntimeError as e:
                raise CarlaRuntimeError(e)
            
            # Spectator following object in simulator tick.
            if self.spectator_following_object is not None:
                try:
                    spectator = self.world.get_spectator()
                    transform = self.spectator_following_object.get_transform()
                    heading = transform.get_forward_vector()
                    up = transform.get_up_vector()
                    cam_loc = transform.location + ((-10) * heading + 3 * up)
                    cam_rot = transform.rotation
                    cam_transform = transform
                    cam_transform.location = cam_loc
                    cam_transform.rotation = cam_rot
                    spectator.set_transform(transform)
                except Exception:
                    traceback.print_exc()
        # Force a GC collect
        gc.collect()
    
    def spectator_track_vehicle(self, vehicle):
        if isinstance(vehicle, CarlaAgentVehicle):
            self.spectator_following_object = vehicle.vehicle
        elif isinstance(vehicle, carla.Vehicle):
            self.spectator_following_object = vehicle
        else:
            raise TypeError(f"Expecting CarlaAgentVehicle or Carla.Vehicle but got {type(vehicle)}")
    
    def spectator_free_mode(self):
        self.spectator_following_object = None
    
    def apply_world_settings(self):
        self.original_settings = self.world.get_settings()
        self.world.unload_map_layer(carla.MapLayer.Foliage)

        # self.traffic_manager.set_hybrid_physics_mode(True)
        # self.traffic_manager.set_hybrid_physics_radius(50.0)

        if self.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)
        
        # # Set traffic light interval
        # for tl in self.world.get_actors().filter('traffic.traffic_light*'):
        #     tl.set_green_time(30)
        #     tl.set_yellow_time(3)
        #     tl.set_red_time(15)
    
    def change_map(self, map_name):
        self.carla.set_timeout(300.0)   # Changing map can take longer time
        self.carla.load_world(map_name)
        self.apply_world_settings()
        self.carla.set_timeout(self.carla_timeout)

    def spawn_traffic(self, number_of_vehicles, number_of_walkers):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')] # T2 bus, buses have issue navigating small intersections
        blueprints = [x for x in blueprints if not x.id.endswith('t2_2021')]
        blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
        blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        blueprints = [x for x in blueprints if not x.id.endswith('european_hgv')]   # truck head
        blueprints = [x for x in blueprints if not x.id.endswith('fusorosa')]   # Fusorosa bus

        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            logging.warning(f'requested {number_of_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points')
            number_of_vehicles = number_of_spawn_points

        spawn_actor = carla.command.SpawnActor
        set_autopilot = carla.command.SetAutopilot
        future_actor = carla.command.FutureActor

        batch = []

        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            batch.append(spawn_actor(blueprint, transform)
                         .then(set_autopilot(future_actor, True, self.traffic_manager_port)))

        for response in self.carla.apply_batch_sync(batch, self.sync):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        all_vehicle_actors = self.world.get_actors(self.vehicles_list)
        for actor in all_vehicle_actors:
            self.traffic_manager.update_vehicle_lights(actor, True)

        percentage_pedestrians_running = 0.0
        percentage_pedestrians_crossing = 0.0

        spawn_points = []

        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # batch = []
        # walker_speed = []
        # for spawn_point in spawn_points:
        #     walker_bp = random.choice(blueprints_walkers)

        #     if walker_bp.has_attribute('is_invincible'):
        #         walker_bp.set_attribute('is_invincible', 'false')

        #     if walker_bp.has_attribute('speed'):
        #         if random.random() > percentage_pedestrians_running:
        #             walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
        #         else:
        #             walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        #     else:
        #         walker_speed.append(0.0)

        #     batch.append(spawn_actor(walker_bp, spawn_point))
        # results = self.carla.apply_batch_sync(batch, self.sync)

        # walker_speed2 = []
        # for i in range(len(results)):
        #     if results[i].error:
        #         logging.error(results[i].error)
        #     else:
        #         self.walkers_list.append({"id": results[i].actor_id})
        #         walker_speed2.append(walker_speed[i])
        # walker_speed = walker_speed2

        # batch = []
        # walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        # for i in range(len(self.walkers_list)):
        #     batch.append(spawn_actor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        # results = self.carla.apply_batch_sync(batch, True)
        # for i in range(len(results)):
        #     if results[i].error:
        #         logging.error(results[i].error)
        #     else:
        #         self.walkers_list[i]["con"] = results[i].actor_id

        # for i in range(len(self.walkers_list)):
        #     self.all_id.append(self.walkers_list[i]["con"])
        #     self.all_id.append(self.walkers_list[i]["id"])
        # self.all_actors = self.world.get_actors(self.all_id)

        # if not self.sync:
        #     self.world.wait_for_tick()
        # else:
        #     self.world.tick()

        # self.world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)

        # for i in range(0, len(self.all_id), 2):
        #     self.all_actors[i].start()
        #     self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
        #     self.all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    def despawn_traffic(self):
        print(f'Despawning traffic')

        self.carla.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        self.carla.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        time.sleep(0.5)

    def spawn_agent_vehicle(self, location=None, rotation=None):
        vehicle = CarlaAgentVehicle(self)
        vehicle.spawn(self.world, location=location, rotation=rotation)
        self.agent_vehicles.append(vehicle)
        return vehicle

    def do_despawn_agent_vehicle(self, vehicle):
        if vehicle in self.agent_vehicles:
            if self.spectator_following_object is vehicle.vehicle:
                self.spectator_following_object = None
            self.agent_vehicles.remove(vehicle)
            vehicle.destroy()

    def despawn_agent_vehicle(self, vehicle):
        self.call_once(self.do_despawn_agent_vehicle, vehicle)

    def despawn_all_agents(self):
        for vehicle in self.agent_vehicles:
            self.despawn_agent_vehicle(vehicle)

    def add_tick_callback(self, tick_callback):
        self.tick_callbacks.append(tick_callback)

    def remove_tick_callback(self, tick_callback):
        if tick_callback in self.tick_callbacks:
            self.tick_callbacks.remove(tick_callback)
    
    def call_once(self, callback, *args, **kwargs):
        self.mq.put(ScheduleCallMessage(callback, *args, **kwargs))

    def close(self):
        self.mq.put_nowait(ExitMessage())
        self.thread.join()


class CarlaRecorder(object):
    def __init__(self, env):
        super().__init__()
        self.save_path = None
        self.env = env
        self.is_recording = False
        self.num_frame_recorded = 0
        self.random_weather = False

    def start_record(self, save_path=None):
        if self.is_recording:
            raise RuntimeError('Recording in progress.')

        self.save_path = save_path
        if self.save_path is None:
            self.save_path = "data/record/" + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + "/"
        
        os.makedirs(self.save_path, exist_ok=True)
        info_data = {
            'sync': self.env.sync,
            'tick_interval': self.env.tick_interval,
            'fixed_delta_seconds': self.env.fixed_delta_seconds,
            'start_frame': self.env.frame,
            'start_timestamp': self.env.timestamp,
        }

        with open(os.path.join(self.save_path, 'info.json'), 'w') as f:
            json.dump(info_data, f, sort_keys=True, indent=4)

        self.is_recording = True
        self.num_frame_recorded = 0
        self.env.add_tick_callback(self.record_callback)

    def stop_record(self):
        self.is_recording = False
        self.env.remove_tick_callback(self.record_callback)

    def get_num_frame_recorded(self):
        return copy.copy(self.num_frame_recorded)

    def record_callback(self):
        frame = self.env.frame
        timestamp = self.env.timestamp

        if self.random_weather and frame % 1000 == 0:
            self.env.world.set_weather(getattr(carla.WeatherParameters, random.choice([
                "Default",
                "ClearNoon",
                "WetCloudyNoon",
                "CloudySunset",
                "HardRainNoon"
            ])))

        if len(self.env.agent_vehicles) == 0:
            return

        for vehicle_id, vehicle in enumerate(self.env.agent_vehicles):
            with vehicle.lock:
                agent_path = os.path.join(self.save_path, "agents", str(vehicle_id))
                if not os.path.exists(agent_path):
                    os.makedirs(agent_path, exist_ok=True)
                    info_data = {
                        'sensors': {},
                    }
                    for sensor_name, sensor in vehicle.sensors.items():
                        info_data['sensors'][sensor_name] = {}
                        info_data['sensors'][sensor_name]['sensor_type'] = sensor.sensor_type
                        if hasattr(sensor, 'transform'):
                            info_data['sensors'][sensor_name]['transform'] = {
                                'location': [
                                    sensor.transform.location.x,
                                    sensor.transform.location.y,
                                    sensor.transform.location.z,
                                ],
                                'rotation': [
                                    sensor.transform.rotation.roll,
                                    sensor.transform.rotation.pitch,
                                    sensor.transform.rotation.yaw,
                                ],
                            }
                        if hasattr(sensor, 'sensor_options'):
                            info_data['sensors'][sensor_name]['sensor_options'] = sensor.sensor_options

                    with open(os.path.join(agent_path, 'sensors.json'), 'w') as f:
                        json.dump(info_data, f)

                for sensor_name, sensor in vehicle.sensors.items():
                    sensor_path = os.path.join(agent_path, sensor_name)
                    os.makedirs(sensor_path, exist_ok=True)
                    data = sensor.fetch()

                    if sensor.sensor_type == 'sensor.camera.rgb':
                        # Save camera data as image
                        im = Image.fromarray(data, mode='RGB')
                        with open(os.path.join(sensor_path, str(frame)+'.png'), 'wb') as f:
                            im.save(f, format='PNG')

                        #rotation, translation, intrinsics = sensor.get_camera_info()
                        # TODO: save rotation, translation and intrinsics
                    elif sensor.sensor_type == 'sensor.camera.semantic_segmentation':
                        features = sensor.visualize_rgb(data)
                        im = Image.fromarray(np.uint8(features))
                        with open(os.path.join(sensor_path, str(frame)+'.png'), 'wb') as f:
                            im.save(f, format='PNG')
                    elif sensor.sensor_type == 'sensor.lidar.ray_cast':
                        # Lidar data are dumped as binary array of floats.
                        lidar_data = data.raw_data
                        with open(os.path.join(sensor_path, str(frame)+'.bin'), 'wb') as f:
                            f.write(lidar_data)
                    elif sensor.sensor_type in [
                        'sensor.other.imu',
                        'sensor.other.gnss',
                        'perception.lane.gt'
                    ]:
                        filepath = os.path.join(sensor_path, 'data.csv')
                        if not os.path.exists(filepath):
                            with open(filepath, 'w') as f:
                                f.write(', '.join(data.keys()) + '\n')
                        with open(filepath, 'a') as f:
                            f.write(', '.join([str(v) for v in data.values()]) + '\n')

                    elif sensor.sensor_type in [
                        'perception.location.gt',
                        'perception.traffic_light.gt',
                        'perception.traffic_sign.gt',
                        'perception.vehicle_bbox.gt',
                        'perception.pedestrian_bbox.gt',
                        'perception.junction.gt',
                    ]:
                        # jsonl file is basically lines of json objects.
                        with open(os.path.join(sensor_path, 'data.jsonl'), 'a+') as f:
                            json.dump(data, f)
                            f.write('\n')
                    else:
                        # For unknown sensors, just dump it into pickle.
                        with open(os.path.join(sensor_path, str(frame)+'.pkl'), 'wb') as f:
                            pickle.dump(data, f)

                # Save the control data
                control = vehicle.control

                control_data = {
                    'frame': frame,
                    'timestamp': timestamp,
                    'throttle': control.throttle,
                    'steer': control.steer,
                    'brake': control.brake,
                    'hand_brake': control.hand_brake,
                    'reverse': control.reverse,
                }

                control_filepath = os.path.join(agent_path, 'control.csv')
                if not os.path.exists(control_filepath):
                    with open(control_filepath, 'w') as f:
                        f.write(', '.join(control_data.keys()) + '\n')

                with open(control_filepath, 'a') as f:
                    f.write(', '.join([str(v) for v in control_data.values()]) + '\n')
        
        self.num_frame_recorded += 1

