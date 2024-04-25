import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class BirdsViewSemanticSegmentation():
    def __init__(self, vehicle, device=None):
        self.vehicle = vehicle
        self.num_classes = 7
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.training = True
        self.result = None
        self.model = BirdsViewSemanticSegmentationModule(num_channels=3*4, num_classes=self.num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.save_path = "data/BirdsViewSemanticSegmentation"
        self.current_loss = 0.0

        self.model.to(self.device)
        os.makedirs(self.save_path, exist_ok=True)

        if os.path.exists(os.path.join(self.save_path, 'model.pth')):
            state_dict = torch.load(os.path.join(self.save_path, 'model.pth'))
            self.model.load_state_dict(state_dict)
            print("Birds View model loaded.")

        self.save_interval = 100
        self.processed_ticks = 0

        self.result = None
    
    def tick(self):
        try:
            input_cameras = [
                self.vehicle.sensors['front_camera'].fetch(),
                self.vehicle.sensors['back_camera'].fetch(),
                self.vehicle.sensors['left_camera'].fetch(),
                self.vehicle.sensors['right_camera'].fetch(),
            ]
            ground_truth = self.vehicle.sensors['birds_view_semantic_camera'].fetch()
            ground_truth = ground_truth.reshape((ground_truth.shape[0], ground_truth.shape[1]))
            input_data = np.concatenate(input_cameras, axis=2, dtype=np.float32) / 255.0
            input_data = torch.tensor(input_data, device=self.device, dtype=torch.float32)
            input_data = input_data.unsqueeze(0).permute(0, 3, 1, 2)
            
            ground_truth = torch.tensor(ground_truth, device=self.device, dtype=torch.long)
            ground_truth = ground_truth.unsqueeze(0)

            pred = self.model(input_data)
            self.result = pred.argmax(dim=1).permute(1, 2, 0).detach().cpu().numpy()
            loss = self.criterion(pred, ground_truth) + dice_loss(F.softmax(pred, dim=1).float(),
                                        F.one_hot(ground_truth, self.model.num_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)
            self.current_loss = loss.item()
            if self.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.processed_ticks % self.save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))
                print('Model saved.')
            self.processed_ticks += 1
            # self.training = True
            # for g in self.optimizer.param_groups:
            #     g['lr'] = 1e-4

        except Exception as e:
            print(f'Error in {self}')
            print(e)
        
    def fetch(self):
        return self.result

class BirdsViewSemanticSegmentationModule(nn.Module):
    """Multi camera fusion U-Net"""
    def __init__(self, num_channels, num_classes, bilinear=False):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(num_channels, 64)
        self.down1 = Downscaling(64, 128)
        self.down2 = Downscaling(128, 256)
        self.down3 = Downscaling(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Downscaling(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Downscaling(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)