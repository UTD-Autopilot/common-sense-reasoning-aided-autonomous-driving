import socket
from contextlib import closing
import numpy as np

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def carla_bbox_to_dict(bbox):
    return {
        'extent': [bbox.extent.x, bbox.extent.y, bbox.extent.z],
        'location': [bbox.location.x, bbox.location.y, bbox.location.z],
        'rotation': [bbox.rotation.roll, bbox.rotation.pitch, bbox.rotation.yaw],
    }

def rpy_to_pyr(rot):
    return np.array(rot)[..., [1, 2, 0]]
