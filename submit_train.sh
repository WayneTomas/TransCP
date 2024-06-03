#!/bin/bash
echo start
source activate /home/mnt/tangwei3/software/miniconda3/envs/pytorch1.7
cd /home/mnt/tangwei3/codes/transCP
export CUDA_VISIBLE_DEVICES=0,1


# train referit
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=29516 train.py --config configs/TransCP_R50_referit.py

# train flickr30k
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29516 train.py --config configs/TransCP_R50_flickr30k.py

# train refcoco (unc)
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=29516 train.py --config configs/TransCP_R50_unc.py

# train refcoco+ (unc+)
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=29516 train.py --config configs/TransCP_R50_unc+.py

# train refcocog (gref)
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=29516 train.py --config configs/TransCP_R50_gref.py --max_query_len 40