import torch
from torch import nn
import torch.nn.functional as F
import timm
import ipdb
from networks.ovit import VisionTransformer, _load_weights


filename = '/home/fpelosin/vit_facil/src/networks/pretrained_weights/augreg_Ti_16-i1k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'


# Custom
ovit = VisionTransformer(embed_dim=192, num_classes=0)
_load_weights(model=ovit, checkpoint_path=filename)
ovit.eval()


# timm
#timm.model.visiontransformer
vit = timm.create_model('vit_tiny_patch16_224', num_classes=0)
timm.models.load_checkpoint(vit, filename)
vit.eval()

inp = torch.rand(1,3,224,224)


for i in range(12):

    vit_chunck = torch.nn.Sequential(vit.patch_embed, vit.pos_drop, vit.blocks[:i])
    ovit_chunck = torch.nn.Sequential(ovit.patch_embed, ovit.pos_drop, ovit.blocks[:i])

    out_vit = vit_chunck(inp)
    out_ovit = ovit_chunck(inp)

    if torch.abs(out_vit - out_ovit).sum() > 1e-8:
        print(f"diff@{i}")
    else:
        print(f"NOT diff@{i}")
    

#print(vit(inp))
#print(ovit(inp))

import ipdb; ipdb.set_trace()

# Sanity check
ovit_state_dict = ovit.state_dict()

for name, param in vit.named_parameters():
    if 'attn' in name or 'mlp' in name:
        ovit_param = ovit_state_dict[name]
        if torch.abs(param.data - ovit_param.data).sum() > 1e-8:
            print(f'--->[ DIFF ] {name} is NOT same')
        else:
            print(f'[ OK ] {name} is same')

