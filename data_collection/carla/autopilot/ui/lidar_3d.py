import datetime
import sys
import threading
import time

import open3d as o3d
import numpy as np
import matplotlib.cm as cm

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class Lidar3D:
    def __init__(self, lidar):
        self.lidar = lidar

        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.run, args=[lidar])
        self.thread.start()

    def run(self, lidar):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)

        self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        self.vis.get_render_option().point_size = 1
        self.vis.get_render_option().show_coordinate_frame = True

        self.point_list = o3d.geometry.PointCloud()
        self.to_o3d(lidar.fetch())

        self.vis.add_geometry(self.point_list)

        while True:
            if self.closed.is_set():
                print("Closing Open3D window...")
                self.vis.destroy_window()
                return

            self.to_o3d(self.lidar.fetch())
            self.vis.update_geometry(self.point_list)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.005)

    def update(self):
        self.to_o3d(self.lidar.fetch())
        self.vis.update_geometry(self.point_list)
        self.vis.poll_events()
        self.vis.update_renderer()

    def to_o3d(self, data):
        data = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        points = data[:, :-1]
        points[:, :1] = -points[:, :1]

        self.point_list.points = o3d.utility.Vector3dVector(points)
        self.point_list.colors = o3d.utility.Vector3dVector(int_color)

    def close(self):
        self.closed.set()






