import datetime
import json
import os
import traceback

import carla
import imgui

import numpy as np

from .window import Window
from .agent_window import AgentWindow
from ..agent import AutonomousVehicle

class CarlaWindow(Window):
    def __init__(self, app):
        super(CarlaWindow, self).__init__()

        self.random_weather = False

        self.recorder_load_path = ""

        self.save_ip = True

        self.app = app
        self.env = None

        self.carla_host = '127.0.0.1'
        if 'carla_host' in self.app.config:
            self.carla_host = self.app.config['carla_host']
        self.carla_port = 2000
        if 'carla_port' in self.app.config:
            self.carla_port = self.app.config['carla_port']

        self.num_to_spawn = 1
        self.num_vehicles = 0
        self.num_walkers = 0
        self.sun_angle = 0

        self.map_name_selector_idx = 0
        self.map_names = ["Town10HD_Opt", "Town11", "Town12", "Town01", "Town02", "Town03", "Town04", "Town05", "Town06HD_Opt", "Town07HD_Opt"]

        self.weather_name = "ClearNoon"

        self.agent_windows = []

        self.recorder = None
        self.recorder_save_path = "data/record/" + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + "/"

    def render(self):
        imgui.begin('Carla', False)

        try:
            if self.env is None:
                imgui.text('Carla Server:')

                changed, self.carla_host = imgui.input_text('Host', self.carla_host, 256)
                changed, self.carla_port = imgui.input_int('Port', self.carla_port)
                clicked = imgui.button('Connect')

                _, self.save_ip = imgui.checkbox("Save IP and port", self.save_ip)

                if clicked:
                    self.on_connect_clicked()

            else:
                imgui.text(f"Traffic Manager Port: {self.env.traffic_manager_port}")
                self.map_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD_Opt", "Town11", "Town12", "Town13", "Town15"]
                clikced, self.map_name_selector_idx = imgui.combo(
                    "Map", self.map_name_selector_idx, self.map_names,
                )
                if clikced:
                    self.on_change_map(self.map_name_selector_idx)

                imgui.new_line()
                imgui.text(f"Number of ego vehicles: {len(self.agent_windows)}")
                imgui.new_line()

                _, self.num_to_spawn = imgui.input_int('Amount to spawn', self.num_to_spawn)
                spawn_vehicle_button = imgui.button('Spawn ego vehicles')
                imgui.new_line()
                spawn_scenario_ego_vehicle_button = imgui.button('Spawn scenario defined ego vehicles')
                imgui.new_line()
                _, self.num_vehicles = imgui.input_int('Amount of vehicles', self.num_vehicles)
                _, self.num_walkers = imgui.input_int('Amount of pedestrians', self.num_walkers)
                generate_traffic_button = imgui.button("Generate traffic")
                imgui.same_line()
                clear_traffic_button = imgui.button("Clear traffic")
                imgui.new_line()

                _, self.weather_name = imgui.input_text('Weather name', self.weather_name, 256)
                change_weather_button = imgui.button("Change weather")
                imgui.new_line()

                _, self.sun_angle = imgui.input_int('Sun angle', self.sun_angle)
                change_sun_button = imgui.button("Change sun angle")
                imgui.new_line()

                if spawn_vehicle_button:
                    prev = len(self.agent_windows)

                    while len(self.agent_windows) < prev + self.num_to_spawn:
                        self.on_spawn_vehicle()
                if spawn_scenario_ego_vehicle_button:
                    self.on_spawn_scenario_ego_vehicle()
                if generate_traffic_button:
                    self.env.spawn_traffic(self.num_vehicles, self.num_walkers)

                if clear_traffic_button:
                    self.env.despawn_traffic()

                if change_weather_button:
                    self.on_change_weather()
                    print("Weather changed to " + self.weather_name)

                if change_sun_button:
                    self.on_change_sun()
                    print("Sun angle changed to " + str(self.sun_angle))

                if self.recorder.is_recording:
                    if imgui.button('Stop Recording'):
                        self.recorder.stop_record()
                    imgui.text("Frames recorded: " + str(self.recorder.get_num_frame_recorded()))
                else:
                    _, self.recorder_save_path = imgui.input_text('Save location', self.recorder_save_path, 256)
                    if imgui.button('Record'):
                        self.recorder.start_record(save_path=self.recorder_save_path)

                _, self.random_weather = imgui.checkbox("Randomize weather every", self.random_weather)

                self.recorder.random_weather = self.random_weather

                if imgui.button('Obstacle scenario town03'):
                    try:
                        from autopilot.environment.carla.scenarios.obstacle_avoidance_town03 import setup_obstacle_avoidance_town03
                        
                        self.env.call_once(setup_obstacle_avoidance_town03, self.env)
                    except Exception as e:
                        traceback.print_exc()

                if imgui.button('Obstacle scenario town04'):
                    try:
                        from autopilot.environment.carla.scenarios.obstacle_avoidance_town04 import setup_obstacle_avoidance_town04
                        
                        self.env.call_once(setup_obstacle_avoidance_town04, self.env)
                    except Exception as e:
                        traceback.print_exc()

                window_to_remove = []
                for idx, agent_window in enumerate(self.agent_windows):
                    if agent_window.open:
                        agent_window.render(idx)
                    else:
                        window_to_remove.append(agent_window)
                
                for w in window_to_remove:
                    self.agent_windows.remove(w)

        finally:
            imgui.end()
    
    def on_change_map(self, map_name_idx):
        self.map_name = self.map_names[map_name_idx]
        self.env.change_map(self.map_name)

    def on_change_sun(self):
        weather = self.env.world.get_weather()
        weather.sun_altitude_angle = self.sun_angle
        self.env.world.set_weather(weather)

    def on_change_weather(self):
        self.env.world.set_weather(getattr(carla.WeatherParameters, self.weather_name))

    def on_spawn_vehicle(self):
        try:
            self.agent_windows.append(AgentWindow(self.env))
        except Exception as e:
            print(traceback.format_exc())

    def on_spawn_scenario_ego_vehicle(self):
        try:
            if self.env.scenario_ego_location is None:
                return
            for ego_location, ego_rotation in zip(self.env.scenario_ego_location, self.env.scenario_ego_rotation):
                agent_vehicle = self.env.spawn_agent_vehicle(location=ego_location, rotation=ego_rotation)
                agent = AutonomousVehicle(env=self.env, vehicle=agent_vehicle)
                self.agent_windows.append(AgentWindow(self.env, agent=agent))
        except Exception as e:
            traceback.print_exc()

    def on_connect_clicked(self):
        try:
            if self.env is None:
                from ..environment.carla.environment import CarlaEnvironment
                from ..environment.carla.environment import CarlaRecorder

                if self.save_ip:
                    self.app.config['carla_host'] = self.carla_host
                    self.app.config['carla_port'] = self.carla_port

                self.env = CarlaEnvironment(
                    carla_host=self.carla_host,
                    carla_port=self.carla_port,
                )
                self.recorder = CarlaRecorder(self.env)
                #print(self.env.carla.get_available_maps())

        except Exception as e:
            print(traceback.format_exc())

    def on_close(self):
        try:
            for agent_window in self.agent_windows:
                agent_window.close()
            self.agent_windows = []
        except Exception as e:
            print(traceback.format_exc())

        if self.env is not None:
            print("Carla backend shutting down...")
            self.env.close()
