from torchvision.models.efficientnet import efficientnet_b0
from torch import nn


def efficientnet_init(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3 and child.out_channels == 32:
            conv2d = nn.Conv2d(2, child.out_channels, kernel_size=child.kernel_size[0], stride=child.stride[0],
                               padding=child.padding[0], dilation=child.dilation[0], groups=child.groups,
                               bias=child.bias is not None,
                               padding_mode=child.padding_mode)
            setattr(module, name, conv2d)
        else:
            efficientnet_init(child)
    return module


def EfficientNetB0Feature(feature_dim=4096):
    model = efficientnet_b0(num_classes=feature_dim)
    return efficientnet_init(model)


def create_model(feature_dim=4096, dtype="stft"):
    return EfficientNetB0Feature(feature_dim)


if __name__ == "__main__":
    model = create_model()
    print(model)
