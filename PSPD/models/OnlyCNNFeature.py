import torch
from torch.nn import Module, Sequential, Conv2d, Conv1d, MaxPool2d, MaxPool1d, Flatten, BatchNorm2d, BatchNorm1d, \
    LazyLinear
import torch.nn.functional as F


class OnlyCNNConvBlock(Module):
    def __init__(self, in_channels=1, out_channels=128, conv_kernel=3, pool_kernel=2, dim=2):
        super(OnlyCNNConvBlock, self).__init__()
        if dim == 2:
            self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, padding=1)
            self.bn = BatchNorm2d(num_features=out_channels)
            self.maxpool = MaxPool2d(kernel_size=pool_kernel)
        else:
            self.conv = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, padding=1)
            self.bn = BatchNorm1d(num_features=out_channels)
            self.maxpool = MaxPool1d(kernel_size=pool_kernel)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        return self.maxpool(x)


class OnlyCNNFeature(Module):
    def __init__(self, feature_dim=4096, dim=2):
        super(OnlyCNNFeature, self).__init__()
        self.conv_block1 = OnlyCNNConvBlock(2, 128, 3, 2, dim)
        self.conv_block2 = Sequential(*[OnlyCNNConvBlock(128, 128, 3, 2, dim) for _ in range(4)])
        self.fc = Sequential(Flatten(), LazyLinear(feature_dim))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc(x)
        return F.relu(x)


def create_model(feature_dim=4096, dtype="stft"):
    return OnlyCNNFeature(feature_dim, 2 if dtype in ["stft","wvd","cwd"] else 1)


if __name__ == "__main__":
    input = torch.randn(128, 2, 64, 64)
    model = create_model()
    output = model(input)
