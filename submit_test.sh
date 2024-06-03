#!/bin/bash
echo start
source activate /home/mnt/tangwei3/software/miniconda3/envs/pytorch1.7
cd /home/mnt/tangwei3/codes/transCP
export CUDA_VISIBLE_DEVICES=0


# test referit
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29516 test.py --config configs/TransCP_R50_referit.py --checkpoint outputs/referit/public/checkpoint_best_acc.pth --batch_size_test 16 --test_split test

# test refcoco (unc)
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29516 test.py --config configs/TransCP_R50_unc.py --checkpoint outputs/unc/public/checkpoint_best_acc.pth --batch_size_test 16 --test_split testB

# test refcoco+ (unc+)
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29516 test.py --config configs/TransCP_R50_unc+.py --checkpoint outputs/unc+/public/checkpoint_best_acc.pth --batch_size_test 16 --test_split val

# test refcocog (gref)
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29516 test.py --config configs/TransCP_R50_gref.py --max_query_len 40 --checkpoint outputs/gref/public/checkpoint_best_acc.pth --batch_size_test 16 --test_split val