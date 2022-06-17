from torch import nn
import torch.nn.functional as F
import timm


# It can handle bsize of 50 to 60

class Timm_Vit_tiny_16_augreg_224(nn.Module):

    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()

        filename = '/home/fpelosin/vit_facil/src/networks/pretrained_weights/augreg_Ti_16-i1k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        #import ipdb; ipdb.set_trace()
        if pretrained:
            self.vit = timm.create_model('vit_tiny_patch16_224', num_classes=0)
            timm.models.load_checkpoint(self.vit, filename)
        else:
            self.vit = timm.create_model('vit_tiny_patch16_224', num_classes=0)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(in_features=192, out_features=num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        h = self.fc(self.vit(x))
        return h


def timm_vit_tiny_16_augreg_224(num_out=100, pretrained=False):
    if pretrained:
        return Timm_Vit_tiny_16_augreg_224(num_out, pretrained)
    else:
        raise NotImplementedError
    assert 1==0, "you should not be here :/"
