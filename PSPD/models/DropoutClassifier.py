from torch import nn


class DropputClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=4):
        super(DropputClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Dropout(0.1), nn.Linear(in_dim, num_classes))

    def forward(self, x):
        return self.fc(x)


def create_model(in_dim=2048, num_classes=4):
    return DropputClassifier(in_dim, num_classes)
