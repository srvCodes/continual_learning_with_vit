# Continual Learning with Vistion Transformers
This repo is the official implementation of our CVPR 2022 workshop paper "Towards Exemplar-Free Continual Learning in Vision Transformers: an Account of Attention, Functional and Weight Regularization".

<div align="center">
<img src="./docs/_static/att_fun.png" width="400px">
</div>

## Running the code

Given below are two examples for attentional and functional variants pooling along the height dimension on ImageNet-100.
1. Attentional variant: 

  ```python
>>> python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name dummy_attentional_exp --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'height'   l
```
  
2. Functional variant:
 ```python
>>> python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name dummy_functional_exp --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'height'   
```
The detailed scripts for our experiments can be found in `scripts/`.

## Cite
If you found our implementation to be useful, feel free to use the citation:
```bibtex
@inproceedings{pelosin2022towards,
  title={Towards exemplar-free continual learning in vision transformers: an account of attention, functional and weight regularization},
  author={Pelosin, Francesco and Jha, Saurav and Torsello, Andrea and Raducanu, Bogdan and van de Weijer, Joost},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3820--3829},
  year={2022}
}
```

This repo is based on [FACIL](https://github.com/mmasana/FACIL).

<div align="center">
<img src="./docs/_static/facil_logo.png" width="100px">
</div>
