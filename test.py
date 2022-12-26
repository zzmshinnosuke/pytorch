from torch.nn import Module
from torch.nn import Linear, LSTM
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
#         self.features = self._vgg_layers(cfg)
        self.test = torch.nn.Parameter(torch.ones(100))

    def _vgg_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x ,kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)
                        ]
                in_channels = x
            
        return nn.Sequential(*layers)

    def forward(self, data):
        out_map = self.features(data)
        return out_map
    
Model = Net()

for name, p in Model.named_parameters():
    print(name)
    print(p.requires_grad)
    print(...)
    