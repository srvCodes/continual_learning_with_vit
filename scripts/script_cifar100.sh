
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

MU=1.0
# LAMBDA=1.0

for LAMBDA in 0.0
do
    for seed in $(seq 1 1 1)                                                                                                                                                                                    
    do                                                                                                                                                                                                          
        python3 -u src/main_incremental.py --datasets cifar100_224 --network OVit_tiny_16_augreg_224 --approach olwf_asym --nepochs 100 --log disk --batch-size 96 --gpu 0 --exp-name 10epochs_sym_pool_layer_6_${SPARSENESS}_lambda_${LAMBDA}_50_cls_6tasks_${seed}_mu_${MU} --after-norm --sym --lr 0.01 --seed ${seed} --lamb $LAMBDA --plast_mu $MU --num-tasks 6 --nc-first-task 50 --lr-patience 15 --int-layer --pool-layers 6     
        python3 -u src/main_incremental.py --datasets cifar100_224 --network OVit_tiny_16_augreg_224 --approach olwf_asym --nepochs 100 --log disk --batch-size 96 --gpu 0 --exp-name 10epochs_sym_pool_layer_1_${SPARSENESS}_lambda_${LAMBDA}_50_cls_6tasks_${seed}_mu_${MU} --after-norm --sym --lr 0.01 --seed ${seed} --lamb $LAMBDA --plast_mu $MU --num-tasks 6 --nc-first-task 50 --lr-patience 15 --int-layer --pool-layers 1      
        python3 -u src/main_incremental.py --datasets cifar100_224 --network OVit_tiny_16_augreg_224 --approach olwf_asym --nepochs 100 --log disk --batch-size 96 --gpu 0 --exp-name 10epochs_sym_pool_layer_12_${SPARSENESS}_lambda_${LAMBDA}_50_cls_6tasks_${seed}_mu_${MU} --after-norm --sym --lr 0.01 --seed ${seed} --lamb $LAMBDA --plast_mu $MU --num-tasks 6 --nc-first-task 50 --lr-patience 15 --int-layer --pool-layers 12       

    done
done 
