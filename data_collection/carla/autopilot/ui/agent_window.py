import random
import copy

import imgui

from .window import Window
from ..agent import AutonomousVehicle

import OpenGL.GL as gl
import weakref
import traceback
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d as o3d

import numpy as np

from ..perception.lift_splat.tools import NormalizeInverse, add_ego
from .lidar_3d import Lidar3D

class AgentWindow(Window):
    def __init__(self, env, agent=None):
        super(AgentWindow, self).__init__()

        self.open = True
        self.env = env
        if agent is None:
            agent_vehicle = self.env.spawn_agent_vehicle()
            self.agent = AutonomousVehicle(env=self.env, vehicle=agent_vehicle)
        else:
            self.agent = agent

        self.sensor_visualizers = {}
        for name, sensor in self.agent.vehicle.sensors.items():
            self.sensor_visualizers[name] = SensorVisualizer(sensor, self.agent)

        self.birds_view_model_texture = create_image_texture()
        self.lift_splat_texture = create_image_texture()

        self.manual_override = False

        # self.fig = plt.figure(figsize=(1, 1), dpi=200)
        # self.fig.tight_layout()

    def render(self, agent_id):
        _, self.open = imgui.begin("Agent " + str(agent_id), True)

        try:
            if self.agent is not None:
                try:
                    if self.env.spectator_following_object is not self.agent.vehicle.vehicle:
                        if imgui.button("Focus"):
                            self.env.spectator_track_vehicle(self.agent.vehicle)
                    else:
                        if imgui.button("Unfocus"):
                            self.env.spectator_free_mode()

                except Exception:
                    #traceback.print_exc()
                    pass

                imgui.text('Agent Information:')

                sensors = imgui.tree_node('Sensors')
                if sensors:
                    try:
                        for name, visualizer in self.sensor_visualizers.items():
                            imgui.text(f'{name}')
                            visualizer.render()
                    except Exception as e:
                        imgui.text(str(e))
                        imgui.text("Sensor may still be loading")
                    finally:
                        imgui.tree_pop()

                if imgui.tree_node('Perception'):
                    try:
                        if hasattr(self.agent, 'localization_model'):
                            imgui.text(str(self.agent.localization_model.fetch()))
                        if hasattr(self.agent, 'lift_splat'):
                            imgui.text("Lift Splat")
                            out = self.agent.lift_splat.fetch()

                            if out is not None:
                                out = np.array(out[0].cpu().detach()) * 255

                                stacked = np.stack((
                                    out.squeeze(0),
                                    np.zeros((200, 200)),
                                    np.zeros((200, 200))),
                                axis=-1)

                                # add ego vehicle
                                stacked[90:110, 95:105, 1] = 128

                                imgui_image_np(self.lift_splat_texture, stacked)

                    except Exception as e:
                        imgui.text(str(e))
                    finally:
                        imgui.tree_pop()

                if imgui.tree_node('Control'):
                    try:
                        changed, autopilot = imgui.checkbox(f'CARLA Autopilot##{id(self.agent)}', self.agent.vehicle.autopilot)
                        if changed:
                            self.agent.vehicle.set_autopilot(autopilot)
                        override_changed, self.manual_override = imgui.checkbox(f'Manual Override##{id(self.agent)}', self.manual_override)
                        if override_changed:
                            # Stop the vehicle when control override changed.
                            self.agent.vehicle.control.throttle = 0.0
                            self.agent.vehicle.control.brake = 1.0
                        if self.manual_override:
                            if imgui.is_key_down(ord('W')):
                                self.agent.vehicle.control.reverse = False
                                self.agent.vehicle.control.brake = 0.0
                                self.agent.vehicle.control.throttle = np.minimum(1.0, self.agent.vehicle.control.throttle+0.1)
                            elif imgui.is_key_down(ord('S')):
                                self.agent.vehicle.control.reverse = True
                                self.agent.vehicle.control.brake = 0.0
                                self.agent.vehicle.control.throttle = np.minimum(1.0, self.agent.vehicle.control.throttle+0.1)
                            else:
                                self.agent.vehicle.control.reverse = False
                                self.agent.vehicle.control.brake = 1.0
                                self.agent.vehicle.control.throttle = 0.0

                            if imgui.is_key_down(ord('A')):
                                self.agent.vehicle.control.steer = np.maximum(-1.0, self.agent.vehicle.control.steer-0.02)
                            elif imgui.is_key_down(ord('D')):
                                self.agent.vehicle.control.steer = np.minimum(1.0, self.agent.vehicle.control.steer+0.02)
                            else:
                                self.agent.vehicle.control.steer = 0.0

                        if self.agent.vehicle.control is not None:
                            visualize_control(self.agent.vehicle.control, allow_modify=False)
                    except Exception as e:
                        imgui.text(str(e))
                    finally:
                        if not self.open:
                            self.on_close()
                        imgui.tree_pop()

                # img = self.agent.birds_view_model.fetch()
                # if img is not None:
                #     img = self.agent.vehicle.sensors['birds_view_semantic_camera'].visualize_rgb(img)
                #     imgui_image_np(self.birds_view_model_texture, img)
                #     imgui.text(f'Loss: {self.agent.birds_view_model.current_loss}')
            if not self.open:
                self.on_close()
        except Exception as e:
            imgui.text(traceback.format_exc())
        finally:
            imgui.end()

    def on_close(self):
        # for sv in self.sensor_visualizers.values():
        #     sv.close()

        if self.agent is not None:
            self.agent.on_close()
            self.agent = None

        if self.birds_view_model_texture is not None:
            gl.glDeleteTextures([self.birds_view_model_texture])
            self.birds_view_model_texture = None

        if self.lift_splat_texture is not None:
            gl.glDeleteTextures([self.lift_splat_texture])
            self.lift_splat_texture = None


def create_image_texture():
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture


def imgui_image_np(texture, image):
    if image is None:
        raise ValueError('Image is None')
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    h, w, _ = image.shape
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image)
    imgui.image(texture, w, h)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


class SensorVisualizer():
    def __init__(self, sensor, agent):
        self.sensor = weakref.proxy(sensor)
        self.texture = None
        self.agent = agent

        camera_types = [
            'sensor.camera.rgb',
            'sensor.camera.semantic_segmentation',
            'sensor.lidar.ray_cast',
        ]
        self.lidar_3d = None

        if self.sensor.sensor_type in camera_types:
            self.texture = create_image_texture()

    def render(self):
        try:
            if self.sensor.sensor_type == 'sensor.camera.rgb':
                if imgui.tree_node(f'Show Image##{id(self.sensor)}'):
                    img = self.sensor.fetch()
                    imgui_image_np(self.texture, img)
                    imgui.tree_pop()
            elif self.sensor.sensor_type == 'sensor.camera.semantic_segmentation':
                if imgui.tree_node(f'Show Semantic Image##{id(self.sensor)}'):
                    ss = self.sensor.fetch()
                    img = self.sensor.visualize_rgb(ss)
                    imgui_image_np(self.texture, img)
                    imgui.tree_pop()
            elif self.sensor.sensor_type == 'sensor.lidar.ray_cast':
                if imgui.tree_node(f'Show Lidar Image##{id(self.sensor)}'):
                    data = self.sensor.fetch()
                    img = self.sensor.lidar_image(data)
                    imgui_image_np(self.texture, img)
                    imgui.tree_pop()
            elif self.sensor.sensor_type == 'perception.traffic_light.gt':
                data = copy.deepcopy(self.sensor.fetch())   # deep copy to workaround memory leak
                imgui.input_text_multiline(f'Data##{id(self.sensor)}', str(data))
            elif self.sensor.sensor_type == 'perception.vehicle_bbox.gt':
                data = copy.deepcopy(self.sensor.fetch())
                imgui.input_text_multiline(f'Data##{id(self.sensor)}', str(data))
            elif self.sensor.sensor_type == 'perception.pedestrian_bbox.gt':
                data = copy.deepcopy(self.sensor.fetch())
                imgui.input_text_multiline(f'Data##{id(self.sensor)}', str(data))
            elif self.sensor.sensor_type == 'perception.lane.gt':
                data = copy.deepcopy(self.sensor.fetch())
                imgui.text(f'has_lane: {data["has_lane"]}')
                imgui.text(f'left_lanes: {data["left_lanes"]}')
                imgui.text(f'right_lanes: {data["right_lanes"]}')
            elif self.sensor.sensor_type == 'perception.junction.gt':
                data = copy.deepcopy(self.sensor.fetch())
                imgui.text(f'Number of nearby junctions: {len(data["junctions"])}')
                imgui.input_text_multiline(f'Data##{id(self.sensor)}', str(data))
            elif self.sensor.sensor_type == 'perception.location.gt':
                data = copy.deepcopy(self.sensor.fetch())
                imgui.input_text_multiline(f'Data##{id(self.sensor)}', str(data))
            else:
                imgui.text(f'Sensor type {self.sensor.sensor_type} not supported.')
                data = copy.deepcopy(self.sensor.fetch())
                imgui.input_text_multiline(f'Data##{id(self.sensor)}', str(data))

        except ReferenceError:
            imgui.text('Sensor has been destroyed')
        
        except Exception:
            traceback.print_exc()

    def close(self):
        self.on_close()

    def on_close(self):
        if self.texture is not None:
            gl.glDeleteTextures([self.texture])
            self.texture = None


def visualize_control(control, allow_modify=False):
    if control is None:
        return

    changed, throttle = imgui.slider_float(
        "throttle", control.throttle,
        min_value=0.0, max_value=1.0
    )
    changed, steer = imgui.slider_float(
        "steer", control.steer,
        min_value=-1.0, max_value=1.0
    )
    changed, brake = imgui.slider_float(
        "brake", control.brake,
        min_value=0.0, max_value=1.0
    )
    _, hand_brake = imgui.checkbox(
        "hand_brake", control.hand_brake,
    )
    _, reverse = imgui.checkbox(
        "reverse", control.reverse,
    )

    if allow_modify:
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        control.hand_brake = hand_brake
        control.reverse = reverse
