from torch import nn
import torch.nn.functional as F
from .ovit import OVisionTransformer, _load_weights, get_attention_list


# It can handle bsize of 50 to 60

class OVit_tiny_16_augreg_224(nn.Module):

    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()

        filename = 'src/networks/pretrained_weights/augreg_Ti_16-i1k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        #import ipdb; ipdb.set_trace()
        self.ovit = OVisionTransformer(embed_dim=192, num_classes=0, num_heads=3)
        if pretrained:
            _load_weights(model=self.ovit, checkpoint_path=filename)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(in_features=192, out_features=num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        h = self.fc(self.ovit(x))
        return h


def ovit_tiny_16_augreg_224(num_out=100, pretrained=False):
    if pretrained:
        return OVit_tiny_16_augreg_224(num_out, pretrained)
    else:
        raise NotImplementedError
    assert 1==0, "you should not be here :/"
