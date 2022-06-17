import torch
from torch import nn
import torch.nn.functional as F
import timm


# It can handle bsize of 50 to 60

class Efficient_net(nn.Module):

    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()

        self.net = torch.hub.load('szq0214/MEAL-V2','meal_v2', 'mealv2_efficientnet_b0', pretrained=False)#.module

        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url('https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV2_EfficientNet_B0_224.pth', map_location="cuda:3")
            state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
            self.net.load_state_dict(state_dict)


        # last classifier layer (head) with as many outputs as classes
        self.net.classifier = nn.Identity()
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.fc = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.head_var = 'fc'
    def forward(self, x):
        h = self.fc(self.net(x))
        return h


def efficient_net(num_out=100, pretrained=False):
    if pretrained:
        return Efficient_net(num_out, pretrained)
    else:
        raise NotImplementedError
    assert 1==0, "you should not be here :/"
