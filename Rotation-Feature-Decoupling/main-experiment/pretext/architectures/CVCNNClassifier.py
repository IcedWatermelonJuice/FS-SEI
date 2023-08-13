from torch.nn import Module, Sequential, LazyLinear


class CVCNNClassifier(Module):
    def __init__(self, opt):
        super(CVCNNClassifier, self).__init__()
        self.fc_classifier = Sequential(LazyLinear(opt['num_classes']))

    def forward(self, feat):
        return self.fc_classifier(feat)


def create_model(opt):
    return CVCNNClassifier(opt)
