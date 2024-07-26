import torch
from torch.nn import Module, Sequential, Conv1d, MaxPool1d, Flatten, BatchNorm1d, LazyLinear
import torch.nn.functional as F


class ComplexConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        # Model components
        self.conv_re = Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv_im = Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shape of x : [batch, channel, axis1]
        x_real = x[:, 0:x.shape[1] // 2, :]
        x_img = x[:, x.shape[1] // 2: x.shape[1], :]
        real = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)
        outputs = torch.cat((real, imaginary), dim=1)
        return outputs


class OnlyCVCNNConvBlock(Module):
    def __init__(self, in_channels=2, out_channels=128, conv_kernel=3, pool_kernel=2):
        super(OnlyCVCNNConvBlock, self).__init__()
        self.conv = ComplexConv(in_channels=in_channels // 2, out_channels=out_channels // 2, kernel_size=conv_kernel,
                                padding=1)
        self.bn = BatchNorm1d(num_features=out_channels)
        self.maxpool = MaxPool1d(kernel_size=pool_kernel)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        return self.maxpool(x)


class OnlyCVCNNFeature(Module):
    def __init__(self, feature_dim=4096):
        super(OnlyCVCNNFeature, self).__init__()
        self.conv_block1 = OnlyCVCNNConvBlock(2, 128, 3, 2)
        self.conv_block2 = Sequential(*[OnlyCVCNNConvBlock(128, 128, 3, 2) for _ in range(4)])
        self.fc = Sequential(Flatten(), LazyLinear(feature_dim))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc(x)
        return F.relu(x)


def create_model(feature_dim=4096):
    return OnlyCVCNNFeature(feature_dim)


if __name__ == "__main__":
    input = torch.randn(128, 2, 64, 64)
    model = create_model()
    output = model(input)
