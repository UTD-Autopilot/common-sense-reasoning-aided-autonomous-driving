import json
import os
import threading
import traceback

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as gl

from .carla_window import CarlaWindow
from .nuscene_window import NuScenesWindow


class AutopilotGUI(object):
    def __init__(self):
        self.data_path = 'data/'
        os.makedirs(self.data_path, exist_ok=True)

        self.config = {}
        try:
            with open(os.path.join(self.data_path, 'config.json'), 'r') as f:
                self.config.update(json.load(f))
        except (FileNotFoundError, IOError, json.decoder.JSONDecodeError):
            pass

        imgui.create_context()
        self.ui_thread = threading.Thread(target=self.ui_loop)

        self.glfw_window = None
        self.default_font = None
        self.color = (1., 1., 1.)

        self.windows = []
        self.windows.append(CarlaWindow(self))
        self.windows.append(NuScenesWindow(self))

        self.closed = False
        self.show_test_window = False

        self.ui_thread.start()

    def ui_loop(self):
        self.glfw_window = self.impl_glfw_init()
        impl = GlfwRenderer(self.glfw_window)

        io = imgui.get_io()
        self.default_font = io.fonts.add_font_from_file_ttf(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Roboto-Regular.ttf"),
            20
        )
        impl.refresh_font_texture()

        style_name = 'style_colors_light'
        self.apply_styles(style_name)

        if style_name == 'style_colors_dark':
            self.color = (0., 0., 0.)

        while not glfw.window_should_close(self.glfw_window):
            glfw.poll_events()
            impl.process_inputs()

            imgui.new_frame()

            io = imgui.get_io()
            io.font_global_scale = 1.0

            imgui.push_font(self.default_font)

            try:
                self.render()
            except Exception as e:
                print(traceback.format_exc())
            imgui.pop_font()

            gl.glClearColor(self.color[0], self.color[1], self.color[2], 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            try:
                imgui.render()
                impl.render(imgui.get_draw_data())
                glfw.swap_buffers(self.glfw_window)
            except Exception as e:
                print(traceback.format_exc())
                self.closed = True

            if self.closed:
                break
        for w in self.windows:
            try:
                if not w.closed:
                    w.close()
            except Exception as e:
                print(e)

        self.save_config()

        impl.shutdown()
        glfw.terminate()

    def render(self):

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu('Settings', True):
                clicked_quit, selected_quit = imgui.menu_item(
                    'Quit', 'Cmd+Q', False, True
                )
                if clicked_quit:
                    self.closed = True

                if imgui.begin_menu('Themes', True):
                    clicked_light, _ = imgui.menu_item('Light', None, False, True)
                    clicked_dark, _ = imgui.menu_item('Dark', None, False, True)

                    if clicked_light:
                        self.apply_styles('style_colors_light')
                        self.color = (1., 1., 1.)
                    if clicked_dark:
                        self.apply_styles('style_colors_dark')
                        self.color = (0., 0., 0.)

                    imgui.end_menu()

                imgui.end_menu()

            if imgui.begin_menu('Debug', True):
                _, self.show_test_window = imgui.checkbox(
                    'Test Window', self.show_test_window
                )

                imgui.end_menu()

            imgui.end_main_menu_bar()

        if self.show_test_window:
            imgui.show_test_window()

        windows = self.windows.copy()
        closed_windows = []
        for w in windows:
            w.render()
            if w.closed:
                closed_windows.append(w)

        for w in closed_windows:
            self.windows.remove(w)

    def impl_glfw_init(self):
        width, height = 1280, 720
        window_name = 'Autopilot'

        if not glfw.init():
            print('Could not initialize OpenGL context')
            exit(1)

        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(
            int(width), int(height), window_name, None, None
        )
        glfw.make_context_current(window)

        if not window:
            glfw.terminate()
            print('Could not initialize Window')
            exit(1)

        return window

    def apply_styles(self, style_name):
        style = imgui.get_style()
        getattr(imgui, style_name)(style)

    def add_window(self, window):
        self.windows.append(window)

    def remove_window(self, window):
        self.windows.remove(window)

    def save_config(self):
        try:
            with open(os.path.join(self.data_path, 'config.json'), 'w') as f:
                jobj = json.dumps(self.config, indent=4)
                f.write(jobj)
        except Exception as e:
            traceback.print_exc()
