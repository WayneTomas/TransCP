# Context Disentangling and Prototype Inheriting for Robust Visual Grounding

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div>
<div align="center">
    <a href='https://github.com/WayneTomas' target='_blank'>Wei Tang<sup>*,1</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=Q-4mZnQAAAAJ&hl=zh-CN' target='_blank'>Liang Li<sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=SVQYcYcAAAAJ' target='_blank'>Xuejing Liu<sup>3</sup></a>&emsp;
    <a href='https://imag-njust.net/lu-jin/' target='_blank'>Lu Jin<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=ByBLlEwAAAAJ' target='_blank'>Jinhui Tang<sup>1</sup></a>&emsp;
    <a href='https://imag-njust.net/zechaoli/' target='_blank'>Zechao Li<sup>&#x2709,1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>Nanjing University of Science and Technology;
    <sup>2</sup>Institute of Computing Technology, Chinese Academy of Science;
    <sup>3</sup>SenseTime Research&emsp;
    </br>
    <sup>&#x2709</sup> Corresponding Author
    
</div>
 
 -----------------

![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWayneTomas%2FTransCP&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Updates
- **28 Nov, 2023**: :boom::boom:  Our paper "Context Disentangling and Prototype Inheriting for Robust Visual Grounding" has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
- **3 june, 2024**: :boom::boom:  The codes has been released.

---
This repository contains the **official implementation** and **checkpoints** of the following paper:

> **Context Disentangling and Prototype Inheriting for Robust Visual Grounding**<br>
> 
>
> **Abstract:** *Visual grounding (VG) aims to locate a specific target in an image based on a given language query. The discriminative information from context is important for distinguishing the target from other objects, particularly for the targets that have the same category as others. However, most previous methods underestimate such information. Moreover, they are usually designed for the standard scene (without any novel object), which limits their generalization to the open-vocabulary scene. In this paper, we propose a novel framework with context disentangling and prototype inheriting for robust visual grounding to handle both scenes. Specifically, the context disentangling disentangles the referent and context features, which achieves better discrimination between them. The prototype inheriting inherits the prototypes discovered from the disentangled visual features by a prototype bank to fully utilize the seen data, especially for the open-vocabulary scene. The fused features, obtained by leveraging Hadamard product on disentangled linguistic and visual features of prototypes to avoid sharp adjusting the importance between the two types of features, are then attached with a special token and feed to a vision Transformer encoder for bounding box regression. Extensive experiments are conducted on both standard and open-vocabulary scenes. The performance comparisons indicate that our method outperforms the state-of-the-art methods in both scenarios. The code is available at https://github.com/WayneTomas/TransCP.*

  
## Todo
1. [x] Update the README.
2. [x] Release the codes.
3. [] Release the checkpoints.
4. [] Release the adapted compared methods codes/ckpts


## Get Start

- [Install](#install)
- [Checkpoint](#checkpoint)

## Install

```shell
git clone https://github.com/WayneTomas/TransCP.git
conda create -n pytorch1.7 python=3.6.13
conda activate pytorch1.7
pip install -r requirements.txt
```

## Dataset
Please follow the instruction of [VLTVG/TransVG](https://github.com/yangli18/VLTVG/blob/master/docs/get_started.md) for dataset preparation.


## Checkpoint
...

## Train

The following is an example of model training on the RefCOCO dataset.
```shell
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29516 train.py --config configs/TransCP_R50_unc.py
```

## Inference

For stanadard scene:
train on RefCOCO train, test on RefCOCO-testB split
```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29516 test.py --config configs/TransCP_R50_unc.py --checkpoint outputs/unc/public/checkpoint_best_acc.pth --batch_size_test 16 --test_split testB
```

For open-vocabulary scene:
train on Ref-Reasoning, test on RefCOCO-testB split
```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29539 --use_env test.py --config configs/TransCP_R50_unc.py --checkpoint outputs/ref_reasoning/publick/checkpoint_best_acc.pth --batch_size_test 16 --test_split testB
```

## Cite

```bibtex
@article{tang2023context,
  title={Context Disentangling and Prototype Inheriting for Robust Visual Grounding},
  author={Wei, Tang and Liang, Li and Xuejing Liu and Lu Jin and Jinhui Tang and Zechao, Li},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  DOI: 10.1109/TPAMI.2023.3339628
  year={2023}
}
```

## Acknowledgement
Part of our code is based on the previous works [DETR](https://github.com/facebookresearch/detr), [TransVG](https://github.com/djiajunustc/TransVG), and [VLTVG](https://github.com/yangli18/VLTVG), thanks for the authors. And we thank for Prof. [Sibei Yang](https://sibeiyang.github.io/dataset/ref-reasoning/) for providing the Ref-Reasoning dataset.
