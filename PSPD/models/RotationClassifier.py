from torch import nn


class RotationClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=4):
        super(RotationClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def create_model(in_dim=2048, num_classes=4):
    return RotationClassifier(in_dim, num_classes)
