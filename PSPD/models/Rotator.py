import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tftb.processing import WignerVilleDistribution


class Rotator(nn.Module):
    def __init__(self, rot_angle=(0, 90, 180, 270), data_type="stft"):
        super(Rotator, self).__init__()
        self.rot_angle = rot_angle
        self.rot_num = len(rot_angle)
        self.data_type = data_type
        self.max_value = 0

    def forward(self, x):
        x_len = len(x)
        x_rot = []
        y_rot = []
        for i in range(self.rot_num):
            x_rot.append(self.to_data_type(self.rot_phase(x, self.rot_angle[i]), self.data_type))
            y_rot.append(torch.tensor([i] * x_len))
        x_rot = torch.cat(x_rot, dim=0)
        y_rot = torch.cat(y_rot, dim=0)
        y_ins = torch.tensor(range(x_len))
        if self.max_value == 0:
            self.max_value = x_rot.max()
        if not (self.max_value == 0 or self.max_value == 1):
            x_rot = x_rot/self.max_value
        return x_rot, y_rot, y_ins, self.rot_num, x_len

    @staticmethod
    def to_data_type(x, dtype="stft"):
        if dtype == "stft":
            return Rotator.to_spect(x)
        elif dtype == "fft":
            return Rotator.to_fft(x)
        elif dtype == "cwd":
            return Rotator.to_wvd(x)
        else:
            return x

    @staticmethod
    def rot_phase(x, angle):
        x = x.clone().cpu()
        if angle == 0:
            return x
        else:
            radians = np.radians(angle)
            cos = np.cos(radians)
            sin = np.sin(radians)
            rotation_matrix = torch.tensor([[cos, -sin], [sin, cos]], dtype=x.dtype)
            return torch.matmul(rotation_matrix, x)

    @staticmethod
    def to_spect(x, window_length=128, hop_length=46):
        spec = torch.fft.fftshift(
            torch.stft(x[:, 0, :] + 1j * x[:, 1, :], n_fft=window_length, hop_length=hop_length,
                       win_length=window_length, center=False,
                       normalized=False, onesided=False), dim=1)
        return torch.cat((spec.abs().log1p().unsqueeze(1), torch.angle(spec).unsqueeze(1)), dim=1)

    @staticmethod
    def to_fft(x):
        y = torch.fft.fft(x[:, 0, :] + 1j * x[:, 1, :])
        return torch.stack([y.real, y.imag], dim=1)

    @staticmethod
    def to_wvd(x):
        device = x.device
        x = x[:, 0, :] + x[:, 1, :] * 1j
        wvd = []
        for i in range(len(x)):
            xi = np.array(x[i])
            spec = WignerVilleDistribution(xi).run()
            wvd.append(Rotator.imresize(spec[0], (256, 256)))
        wvd = torch.tensor(np.array(wvd)).to(device)
        return torch.cat((wvd.abs().log1p().unsqueeze(1), torch.angle(wvd).unsqueeze(1)), dim=1)

    @staticmethod
    def imresize(img, size):
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize(size, Image.BICUBIC)  # »òÕßÊ¹ÓÃ Image.NEAREST¡¢Image.BILINEARµÈ²åÖµ·½·¨
        return np.array(img_resized)


def create_model(rot_angle=(0, 90, 180, 270), data_type="stft"):
    return Rotator(rot_angle, data_type)
