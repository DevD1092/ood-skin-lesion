import torch
import torchvision.models as models
import torch.nn as nn
import timm

class SimpleCNN(nn.Module):
    def __init__(self, model,output_size, pretrained=True, loss='Softmax'):
        super(SimpleCNN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        self.loss = loss
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, output_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        if self.loss == 'Softmax':
            return out
        elif self.loss == 'GCPLoss':
            return features, out