import torch.nn as nn

class Network(nn.Module):
    def __init__(self, nchannels, img_dim, nclasses, width, nhidden, dropout, conv_ks, m_ks, conv_padding, nconv, fc_width):
        super(Network, self).__init__()
        self.features = self._make_layers(nchannels)
        self.classifier = nn.Sequential(
            nn.Linear(width//4 * width//4 * 128, nhidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(nhidden, nclasses),
        )

    def _make_layers(self, nchannels):
        layers = []
        layers += [nn.Conv2d(nchannels, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x