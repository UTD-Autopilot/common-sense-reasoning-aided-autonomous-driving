import datetime
import json
import os
import threading
import traceback

import carla
import imgui

from .window import Window
from .agent_window import AgentWindow

class NuScenesWindow(Window):
    def __init__(self, app):
        super(NuScenesWindow, self).__init__()

        self.app = app
        self.env = None
        self.agent_window = None

        self.nuscenes_path = ''
        if 'nuscenes_path' in self.app.config:
            self.nuscenes_path = self.app.config['nuscenes_path']
        self.nuscenes_loading = False

        self.current_scene_idx = 0

    def render(self):
        imgui.begin('NuScenes', False)
        try:
            if self.env is None:
                if not self.nuscenes_loading:
                    imgui.text('Click to load NuScenes dataset:')
                    _, self.nuscenes_path = imgui.input_text('NuScenes path', self.nuscenes_path, 1024)
                    clicked = imgui.button('Load')

                    if clicked and self.nuscenes_path is not None and len(self.nuscenes_path) > 0:
                        self.nuscenes_loading = True
                        threading.Thread(target=self.run_load_nuscenes, daemon=True).start()
                        self.app.config['nuscenes_path'] = self.nuscenes_path
                else:
                    imgui.text('Loading NuScenes dataset...')
            else:
                # Scenes
                scenes = self.env.get_scenes()
                clicked, self.current_scene_idx = imgui.combo("Scene", self.current_scene_idx, [scene['name'] for scene in scenes])
                if clicked:
                    self.env.change_scene(self.current_scene_idx)
                
                imgui.text('Playback:')
                scene_length = scenes[self.current_scene_idx]['nbr_samples']
                changed, current_sample_number = imgui.slider_int('Location', self.env.current_sample_number, min_value=0, max_value=scene_length-1, format='%d')
                if changed:
                    self.env.seek_by_sample_number(current_sample_number)
                _, self.env.paused = imgui.checkbox('Paused', self.env.paused)

                imgui.text(f'timestamp: {self.env.timestamp}')
                
                imgui.text('Description:')
                scene_description = scenes[self.current_scene_idx]['description']
                imgui.push_text_wrap_position()
                imgui.text(scene_description)
                imgui.pop_text_wrap_position()
                
                # AgentWindow must be created on UI thread.
                if self.agent_window is None:
                    self.agent_window = AgentWindow(self.env)
                if self.agent_window.open:
                    self.agent_window.render(0)

        finally:
            imgui.end()
    
    def run_load_nuscenes(self):
        from ..environment.nuscenes.environment import NuScenesEnvironment
        version = os.path.basename(os.path.normpath(self.nuscenes_path))
        self.env = NuScenesEnvironment(version, self.nuscenes_path)
        self.nuscenes_loading = False
