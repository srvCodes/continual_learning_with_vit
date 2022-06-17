# Continual Learning with Vistion Transformers
This repo hosts the official implementation of our CVPR 2022 workshop paper [Towards Exemplar-Free Continual Learning in Vision Transformers: an Account of Attention, Functional and Weight Regularization](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Pelosin_Towards_Exemplar-Free_Continual_Learning_in_Vision_Transformers_An_Account_of_CVPRW_2022_paper.html).

TLDR; We introduce attentional and functional variants for asymmetric and symmetric Pooled Attention Distillation (PAD) losses:
<div align="center">
<img src="./docs/_static/att_fun.png" width="300px">
</div>

## Running the code

<div align="center">
<img src="./docs/_static/asym_illustration(1).png" width="500px">
</div>


Given below are two examples for the asymmetric attentional and functional variants pooling along the height dimension on ImageNet-100.
1. Attentional variant: 

  ```python
>>> python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name dummy_attentional_exp --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'height'   l
```
  
2. Functional variant:
 ```python
>>> python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name dummy_functional_exp --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'height'   
```

The corresponding runs for symmetric variants would then be:
1. Attentional variant: 

  ```python
>>> python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name dummy_attentional_exp --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'height' --sym 
```
  
2. Functional variant:
 ```python
>>> python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name dummy_functional_exp --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'height' --sym 
```

Current available approaches with Vision Transformers include:
<div align="center">
<p align="center"><b>
  EWC • Finetuning • LwF • PathInt 
</b></p>
</div>

The detailed scripts for our experiments can be found in `scripts/` repo.

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
## Acknowledgement
This repo is based on [FACIL](https://github.com/mmasana/FACIL).

<div align="center">
<img src="./docs/_static/facil_logo.png" width="100px">
</div>
