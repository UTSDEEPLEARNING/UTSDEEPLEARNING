import torch.nn as nn

def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))

class ImageCNN(nn.Module):
    def __init__(self, n_class=7):
        super().__init__()
        self.body = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                  conv_block(64, 128), conv_block(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, n_class)
    def features(self, x):
        return self.pool(self.body(x)).flatten(1)
    def forward(self, x):
        return self.head(self.features(x))

class SensorCNN(nn.Module):
    def __init__(self, in_ch=6, n_class=7):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.body = nn.Sequential(blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, n_class)
    def features(self, x):
        return self.pool(self.body(x)).flatten(1)
    def forward(self, x):
        return self.head(self.features(x))
