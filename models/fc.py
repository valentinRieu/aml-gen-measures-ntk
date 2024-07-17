import torch.nn as nn

class Network(nn.Module):
    def __init__(self, nchannels, img_dim, nclasses, width, nhidden, dropout, init_method):
        super(Network, self).__init__()
        modules = [
            nn.Linear( nchannels * img_dim * img_dim, width ),
            nn.ReLU(inplace=True)
        ]
        for i in range(nhidden):
            if dropout > 0:
                modules.append(nn.Dropout(p = dropout, inplace=False))
            
            modules.append(nn.Linear(width, width))
            modules.append(nn.ReLU(inplace=True))
        
        modules.append(nn.Linear(width, nclasses))
        self.classifier = nn.Sequential(*modules)

        self._initialize_weights(init_method)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self, init_method):
        initializer = None
        if init_method == 'xavier_uniform':             # a = 1 * sqrt(6/(fan_in + fan_out)), U(-a, a)
            initializer = nn.init.xavier_uniform_ 
        elif init_method == 'xavier_normal':              # std = sqrt(2/(fan_in +)), N(0, std)
            initializer = nn.init.xavier_normal_
        elif init_method == 'kaiming_uniform':            # a = sqrt(3/fan_in), U(-a, a)
            initializer = nn.init.kaiming_uniform_
        elif init_method == 'kaiming_normal':             # std = 1/sqrt(fan_in), N(0, std²)
            initializer = nn.init.kaiming_normal_
        elif init_method == 'uniform':                    # U(0, 1)
            initializer = nn.init.uniform_
        else:        # No other solutions: either 'normal' or error in the output => N(0, 1) by default 
            initializer = nn.init.normal_
        self._apply_initializer(initializer)
        
    def _apply_initializer(self, initializer):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initializer(m.weight)
                if m.bias is None:
                    nn.init.zeros_(m.bias)
        
                
