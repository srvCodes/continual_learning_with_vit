from torch import nn
import torch.nn.functional as F
from .vit_original import VisionTransformer, _load_weights
from transformers import GPT2Model


# It can handle bsize of 50 to 60

class FPT(nn.Module):

    def __init__(self, num_classes=100, pretrained=False):
        super().__init__(

        if not pretrained:
            asser 1==0, 'cannot run without pretrained'

        self.patch_size = 8
        self.input_dim = 3 * self.patch_size**2

        self.in_net = nn.Linear(input_dim, 768, bias=True)
        self.fpt = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        self.head_var = 'fc'




    def forward(self, x):

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.in_net(x)
        with torch.no_grad():
            x = self.fpt(x)
        
        h = self.fc(x)

        return h


def fPT(num_out=100, pretrained=False):
    if pretrained:
        return FPT(num_out, pretrained)
    else:
        raise NotImplementedError
    assert 1==0, "you should not be here :/"
