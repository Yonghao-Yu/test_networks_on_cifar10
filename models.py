import torch.nn as nn
from torchvision.models import resnet34, vgg19


class MLP(nn.Module):
    def __init__(self, img_size: int = 32, in_chans: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_size * img_size * in_chans, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 248),
            nn.LayerNorm(248),
            nn.ReLU(),
            nn.Linear(248, 10)
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc(x)
        return x


def build_model(model: str, **kwargs):
    if model == 'resnet34':
        return resnet34(**kwargs)
    elif model == 'vgg19':
        return vgg19(**kwargs)
    elif model == 'mlp':
        return MLP()
    else:
        assert False, 'model name error'
