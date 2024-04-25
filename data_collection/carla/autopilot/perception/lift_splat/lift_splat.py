import cv2
import torchvision
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from autopilot.environment.carla import RGBCamera, LiDAR
from autopilot.perception.lift_splat.models import compile_model
from autopilot.perception.lift_splat.pointpillars.pointpillars import PFN
from autopilot.perception.lift_splat.tools import gen_dx_bx, img_transform, NormalizeInverse


class LiftSplat:
    def __init__(self, vehicle, model_path="./autopilot/perception/lift_splat/models/model525000.pt", device=None):
        self.vehicle = vehicle
        self.device = device
        self.result = None

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.grid_conf = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [4.0, 45.0, 1],
        }

        self.data_aug_conf = {
            'resize_lim': (0, 0),
            'final_dim': (128, 352),
            'rot_lim': (0, 0),
            'H': 128, 'W': 352,
            'rand_flip': False,
            'bot_pct_lim': (0.0, 0.0),
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'Ncams': 6,
        }

        self.input_sensors = [
            self.vehicle.sensors['left_front_camera'],
            self.vehicle.sensors['front_camera'],
            self.vehicle.sensors['right_front_camera'],
            self.vehicle.sensors['left_back_camera'],
            self.vehicle.sensors['back_camera'],
            self.vehicle.sensors['right_back_camera'],
            self.vehicle.sensors['lidar'],
        ]

        self.denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))

        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ))

        self.model = self.load_model(model_path)
        self.pillar_feature_net = PFN()

    def load_model(self, model_path):
        model = compile_model(self.grid_conf, self.data_aug_conf, outC=1)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(self.device)

        return model

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']

        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        new_w, new_h = resize_dims

        crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * new_h) - fH
        crop_w = int(max(0, new_w - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        flip = False
        rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def tick(self):
        try:
            imgs = []
            points = []
            rots = []
            trans = []
            intrins = []
            post_rots = []
            post_trans = []

            for sensor in self.input_sensors:
                data = sensor.fetch()
                if data is None:
                    return

                if isinstance(sensor, RGBCamera):
                    rot, tran, intrin = sensor.get_camera_info()

                    # data augmentation
                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)
                    resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

                    img, post_rot2, post_tran2 = img_transform(data, post_rot, post_tran,
                                                               resize=resize,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate, )

                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    imgs.append(self.normalize_img(img))
                    rots.append(rot.tolist())
                    trans.append(tran)
                    intrins.append(intrin.tolist())
                    post_trans.append(post_tran.tolist())
                    post_rots.append(post_rot.tolist())
                elif isinstance(sensor, LiDAR):
                    data = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
                    data = np.reshape(data, (int(data.shape[0] / 4), 4))

                    points.append(data)

                    # code to convert point cloud to pillars
                    # pillars = self.pillar_feature_net(torch.tensor(np.array([data])))
                    # print(pillars.shape)
                    # cv2.imshow("feature net", np.array(pillars[0][0].detach()))
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

            imgs = torch.stack([torch.stack(imgs)]).to(self.device).float()
            points = torch.tensor(np.array(points)).to(self.device).float()
            trans = torch.stack([torch.tensor(trans)]).to(self.device).float()
            rots = torch.stack([torch.tensor(rots)]).to(self.device).float()
            intrins = torch.stack([torch.tensor(intrins)]).to(self.device).float()
            post_rots = torch.stack([torch.tensor(post_rots)]).to(self.device).float()
            post_trans = torch.stack([torch.tensor(post_trans)]).to(self.device).float()

            # code to visualize frustum projections (saved in frustums.jpg)

            # img_pts = self.model.get_geometry(rots, trans, intrins, post_rots, post_trans)
            # plt.xlim((-100, 100))
            # plt.ylim((-100, 100))
            # for i, img in enumerate(imgs[0]):
            #     plt.plot(img_pts[0, i, :, :, :, 0].view(-1), img_pts[0, i, :, :, :, 1].view(-1), '.',
            #              label=self.data_aug_conf['cams'][i].replace('_', ' '))
            #
            # plt.legend(loc='upper right')
            # name = 'frustums.jpg'
            # print('saving', name)
            # plt.savefig(name)
            # plt.clf()

            self.result = self.model(imgs,
                                     rots,
                                     trans,
                                     intrins,
                                     post_rots,
                                     post_trans).sigmoid()

        except Exception as e:
            raise Exception

    def fetch(self):
        return self.result
