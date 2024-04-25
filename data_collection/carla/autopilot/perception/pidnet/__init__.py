import os
import torch

from .pidnet import PIDNet

class PIDNetSemanticSegmentation():
    def __init__(self, vehicle, device=None):
        self.vehicle = vehicle
        self.num_classes = 7
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.save_path = "data/PIDNetSemanticSegmentation"

        self.model = PIDNet(m=3, n=4, num_classes=self.num_classes, planes=64, ppm_planes=112, head_planes=256, augment=True)
        self.model.to(self.device)
        os.makedirs(self.save_path, exist_ok=True)
        if os.path.exists(os.path.join(self.save_path, 'model.pth')):
            state_dict = torch.load(os.path.join(self.save_path, 'model.pth'))
            self.model.load_state_dict(state_dict)
            print("PIDNet model loaded.")

        self.input_sensors = [
            self.vehicle.sensors['left_front_camera'],
            self.vehicle.sensors['front_camera'],
            self.vehicle.sensors['right_front_camera'],
            self.vehicle.sensors['left_back_camera'],
            self.vehicle.sensors['back_camera'],
            self.vehicle.sensors['right_back_camera'],
            self.vehicle.sensors['lidar'],
        ]

        self.result = None
    
    def tick(self):
        try:
            self.result = {}
            for sensor in self.input_sensors:
                data = sensor.fetch()
                if data is None:
                    return
                x = torch.tensor(data, device=self.device, dtype=torch.float32)
                x = x.unsqueeze(0).permute(0, 3, 1, 2)
                pred = self.model(x)
                self.result[sensor] = pred.argmax(dim=1).permute(1, 2, 0).detach().cpu().numpy()
        except Exception as e:
            print(f'Error in {self}')
            print(e)
    
    def fetch(self):
        return self.result