import torch
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0 as shufflenet
from torch import nn


def shufflenet_init(module, in_ch=1):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3 and child.out_channels == 24:
            conv2d = nn.Conv2d(in_ch, child.out_channels, kernel_size=child.kernel_size[0], stride=child.stride[0],
                               padding=child.padding[0], dilation=child.dilation[0], groups=child.groups,
                               bias=child.bias is not None,
                               padding_mode=child.padding_mode)
            setattr(module, name, conv2d)
        else:
            shufflenet_init(child, in_ch)
    return module


def shufflenet_init_1d(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            conv1d = nn.Conv1d(2 if (child.in_channels == 3 and child.out_channels == 64) else child.in_channels,
                               child.out_channels, kernel_size=child.kernel_size[0], stride=child.stride[0],
                               padding=child.padding[0], dilation=child.dilation[0], groups=child.groups,
                               bias=child.bias is not None, padding_mode=child.padding_mode)
            setattr(module, name, conv1d)
        elif isinstance(child, nn.BatchNorm2d):
            bn1d = nn.BatchNorm1d(child.num_features, eps=child.eps, momentum=child.momentum, affine=child.affine,
                                  track_running_stats=child.track_running_stats)
            setattr(module, name, bn1d)
        elif isinstance(child, nn.MaxPool2d):
            maxpool1d = nn.MaxPool1d(kernel_size=child.kernel_size, stride=child.stride, padding=child.padding,
                                     dilation=child.dilation, return_indices=child.return_indices,
                                     ceil_mode=child.ceil_mode)
            setattr(module, name, maxpool1d)
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            avgpool1d = nn.AdaptiveAvgPool1d(output_size=child.output_size[0])
            setattr(module, name, avgpool1d)
        else:
            shufflenet_init_1d(child)
    return module


def ShuffleNetFeature(feature_dim=4096):
    model = shufflenet(num_classes=feature_dim)
    return shufflenet_init(model,2)


def create_model(feature_dim=4096, dtype="stft"):
    return ShuffleNetFeature(feature_dim)


if __name__ == "__main__":
    model = create_model()
    print(model)
    x = torch.rand(32, 2, 256, 256)
    print(x.shape)
    y = model(x)
    print(y.shape)
