#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running approach: $1"
else
    echo "No approach has been assigned."
fi
if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$4" != "" ]; then
    RESULTS_DIR=$4
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

GPU=0
NUM_TASKS=15
NC_FIRST_TASK=20
NEPOCHS=100
for seed in $(seq 1 1 1)                                                                                                                                                                                  
do
    # attentional runs
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sym_pixel_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'pixel' 
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asym_pixel_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'pixel'
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sym_spatial_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'spatial'   
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asym_spatial_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'spatial'   
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sym_width_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'width' 
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asym_width_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'width'
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sym_height_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'height'   
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asym_height_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'height'   l
    # functional runs
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sympost_pixel_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'pixel'
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asympost_pixel_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'pixel'
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sympost_spatial_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'spatial'  
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asympost_spatial_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'spatial'   
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sympost_width_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'width'  
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asympost_width_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'width'
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sympost_height_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0  --pool-along 'height'  
    python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asympost_height_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0 --pool-along 'height'   

    # python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_lwf_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --after-norm --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 0
    # python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach finetuning --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_finetuning_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20      
    # python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach ewc --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name  ${NEPOCHS}_epochs_small_vit_ewc_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20
    # python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach path_integral --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name  ${NEPOCHS}_epochs_small_vit_path_integral_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20
    # python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sympost_pixel_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0     
    # python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asympost --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asympost_mu_1.0_lwf_lambda_1.0_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --lamb 1.0 --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu 1.0    
    # for MU in 1.0 0.7 0.6
    # do
    #     for LAMBDA in 1.0 0.6
    #     do
    #         python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --sym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_sym_mu_${MU}_lwf_lambda_${LAMBDA}_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --after-norm --lr 0.01 --seed ${seed} --lamb $LAMBDA --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu $MU     
    #         python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_asym --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_asym_mu_${MU}_lwf_lambda_${LAMBDA}_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --after-norm --lr 0.01 --seed ${seed} --lamb $LAMBDA --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --plast_mu $MU  
    #         python3 -u src/main_incremental.py --datasets imagenet_32_reduced --network Early_conv_vit --approach olwf_jsd --nepochs $NEPOCHS --log disk --batch-size 1024 --gpu $GPU --exp-name ${NEPOCHS}_epochs_small_vit_jsd_mu_${MU}_lambda_${LAMBDA}_${NC_FIRST_TASK}_cls_${NUM_TASKS}_tasks_${seed} --lr 0.01 --seed ${seed} --plast_mu $MU --num-tasks $NUM_TASKS --nc-first-task $NC_FIRST_TASK --lr-patience 20 --lamb $LAMBDA              
    #     done
    # done
done
