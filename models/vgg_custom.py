import torch.nn as nn

class Network(nn.Module):
    def __init__(self, nchannels, img_dim, nclasses, width, nhidden, dropout, conv_ks, m_ks, conv_padding, nconv, fc_width):
        super(Network, self).__init__()
        self.features = self.make_layers(nchannels, width, nhidden, conv_ks, m_ks, conv_padding, nconv)
        modules = []
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        modules += [
            nn.Linear(width * (img_dim // (m_ks ** nhidden)) ** 2, 4096),
            nn.ReLU(True)
        ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        
        
        modules += [
            nn.Linear(fc_width, fc_width),
            nn.ReLU(True),
            nn.Linear(fc_width, nclasses)
        ]
        self.classifier = nn.Sequential(*modules)

    def make_layers(self, nchannels, width, nhidden, conv_ks, m_ks, conv_padding, nconv):
        layers = []
        in_channels = nchannels
        for _ in range(nhidden):
            for _ in range(nconv):
                layers += [nn.Conv2d(in_channels, width, kernel_size=conv_ks, padding=conv_padding), nn.ReLU(inplace=True)]
                in_channels = width
            layers += [nn.MaxPool2d(kernel_size=m_ks, stride=m_ks)]
            width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x