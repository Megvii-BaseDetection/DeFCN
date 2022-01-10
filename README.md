# End-to-End Object Detection with Fully Convolutional Network

![GitHub](https://img.shields.io/github/license/Megvii-BaseDetection/DeFCN)

This project provides an implementation for "[End-to-End Object Detection with Fully Convolutional Network](https://arxiv.org/abs/2012.03544)" on PyTorch.

Experiments in the paper were conducted on the internal framework, thus we reimplement them on [cvpods](https://github.com/Megvii-BaseDetection/cvpods) and report details as below.

![](./pipeline.png)

## Requirements
* [cvpods](https://github.com/Megvii-BaseDetection/cvpods)
* scipy >= 1.5.4

## Get Started

* install cvpods locally (requires cuda to compile)
```shell

python3 -m pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
python3 -m pip install -e cvpods

# Or,
pip install -r requirements.txt
python3 setup.py build develop
```

* prepare datasets
```shell
cd /path/to/cvpods
cd datasets
ln -s /path/to/your/coco/dataset coco
```

* Train & Test
```shell
git clone https://github.com/Megvii-BaseDetection/DeFCN.git
cd DeFCN/playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms  # for example

# Train
pods_train --num-gpus 8

# Test
pods_test --num-gpus 8 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional

# Multi node training
## sudo apt install net-tools ifconfig
pods_train --num-gpus 8 --num-machines N --machine-rank 0/1/.../N-1 --dist-url "tcp://MASTER_IP:port"

```

## Results on COCO2017 val set

| model | assignment | with NMS | lr sched. | mAP | mAR | download |
|:------|:----------:|:--------:|:---------:|:---:|:---:|:--------:|
| [FCOS](./playground/detection/coco/fcos.res50.fpn.coco.800size.3x_ms) | one-to-many | Yes | 3x + ms | 41.4 | 59.1 | [weight](https://drive.google.com/file/d/1j9FmyQQxB2g3J4M7F5DubBtW_7qXHiMv/view?usp=sharing) \| [log](https://drive.google.com/file/d/18RK2jZd7g198hAeAz80BsD_6cF8aY1mb/view?usp=sharing) |
| [FCOS baseline](./playground/detection/coco/fcos.res50.fpn.coco.800size.3x_ms.wo_ctrness) | one-to-many | Yes | 3x + ms | 40.9 | 58.4 | [weight](https://drive.google.com/file/d/1diZQFuJQR6XzPXJsyh1zrRuFYjbqKZ9l/view?usp=sharing) \| [log](https://drive.google.com/file/d/1P1ouRHmSMB4-WZ_yu46lU3kVXlDQAkdE/view?usp=sharing) |
| [Anchor](./playground/detection/coco/anchor.res50.fpn.coco.800size.3x_ms) | one-to-one | No | 3x + ms | 37.1 | 60.5 | [weight](https://drive.google.com/file/d/1ZVAZPoOlwtNVlxkaKEFWPrkH57nRpuKr/view?usp=sharing) \| [log](https://drive.google.com/file/d/1CVTcCJvLfPPCDN2rIhk8gX8vp98oQidM/view?usp=sharing) |
| [Center](./playground/detection/coco/center.res50.fpn.coco.800size.3x_ms) | one-to-one | No | 3x + ms | 35.2 | 61.0 | [weight](https://drive.google.com/file/d/1TgNFHMs9uxjTrMMRTSXarwVWZOkX53av/view?usp=sharing) \| [log](https://drive.google.com/file/d/1zcnQTQaOXPLLoHy9lHwfFdESxhIkqD1R/view?usp=sharing) |
| [Foreground Loss](./playground/detection/coco/loss.res50.fpn.coco.800size.3x_ms) | one-to-one | No | 3x + ms | 38.7 | 62.2 | [weight](https://drive.google.com/file/d/1rTsXbEC5Tj8kwXdjTuHYcfoap4TsnkXV/view?usp=sharing) \| [log](https://drive.google.com/file/d/1EAMPnK7s0TabKKzZhWjALsY1Hege4pFx/view?usp=sharing) |
| [POTO](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms) | one-to-one | No | 3x + ms | 39.2 | 61.7 | [weight](https://drive.google.com/file/d/1mlk5dxc34PyXMajinlF_zWXxs84Z28MH/view?usp=sharing) \| [log](https://drive.google.com/file/d/1v4TBsbExylfgM7GfGh02vks8AnwSbPDI/view?usp=sharing) |
| [POTO + 3DMF](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf) | one-to-one | No | 3x + ms | 40.6 | 61.6 | [weight](https://drive.google.com/file/d/1yUzhK_wtzr4_hqi_WT3YpDryGn_rrltU/view?usp=sharing) \| [log](https://drive.google.com/file/d/1ik5JnVLIzmuYlbCkq_MTEDrd2jWoNprV/view?usp=sharing) |
| [POTO + 3DMF + Aux](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf.aux) | mixture\* | No | 3x + ms | 41.4 | 61.5 | [weight](https://drive.google.com/file/d/1bxpmTzVzCkV6BHzca_TVWo3pTOEZMAFS/view?usp=sharing) \| [log](https://drive.google.com/file/d/12LTwMJ3zuBYVa7K0OA0ZRTfC1kxianjW/view?usp=sharing) |

\* We adopt a one-to-one assignment in POTO and a one-to-many assignment in the auxiliary loss, respectively.

- `2x + ms` schedule is adopted in the paper, but we adopt `3x + ms` schedule here to achieve higher performance.
- It's normal to observe ~0.3AP noise in POTO.

## Results on CrowdHuman val set

| model | assignment | with NMS | lr sched. | AP50 | mMR | recall | download |
|:------|:----------:|:--------:|:---------:|:----:|:---:|:------:|:--------:|
| [FCOS](./playground/detection/crowdhuman/fcos.res50.fpn.crowdhuman.800size.30k) | one-to-many | Yes | 30k iters | 86.1 | 54.9 | 94.2 | [weight](https://drive.google.com/file/d/1qf34m13kniTK2fo2o8etjMfocezSyosQ/view?usp=sharing) \| [log](https://drive.google.com/file/d/1DgZbvawWGX7rBonS8WgcByIGn7nLNrmA/view?usp=sharing) |
| [ATSS](./playground/detection/crowdhuman/atss.res50.fpn.crowdhuman.800size.30k) | one-to-many | Yes | 30k iters | 87.2 | 49.7 | 94.0 | [weight](https://drive.google.com/file/d/1J30DVItPgLVg9_ps-NdCXWYqaV0PvwAq/view?usp=sharing) \| [log](https://drive.google.com/file/d/1jdL2v_A_fhU6GjYBOzT80ps5CZEZBtx5/view?usp=sharing) |
| [POTO](./playground/detection/crowdhuman/poto.res50.fpn.crowdhuman.800size.30k) | one-to-one | No | 30k iters | 88.5 | 52.2 | 96.3 | [weight](https://drive.google.com/file/d/1mbP0mmHpva30BcQIxY84XhEMsTGwi-ze/view?usp=sharing) \| [log](https://drive.google.com/file/d/1dmn2ENMkfNXaQUaruSR9Pu1QAAOAhlEC/view?usp=sharing) |
| [POTO + 3DMF](./playground/detection/crowdhuman/poto.res50.fpn.crowdhuman.800size.30k.3dmf) | one-to-one | No | 30k iters | 88.8 | 51.0 | 96.6 | [weight](https://drive.google.com/file/d/1d_Z6g54RTIVYHzaUrEogmL3gId2PTBSb/view?usp=sharing) \| [log](https://drive.google.com/file/d/12G-1nm34DjH2xJGRMsiV8OYIZzWooFkt/view?usp=sharing) |
| [POTO + 3DMF + Aux](./playground/detection/crowdhuman/poto.res50.fpn.crowdhuman.800size.30k.3dmf.aux) | mixture\* | No | 30k iters | 89.1 | 48.9 | 96.5 | [weight](https://drive.google.com/file/d/1P5uWt4kjQnm-P_WC0MzqLC5TWbIH62UY/view?usp=sharing) \| [log](https://drive.google.com/file/d/1sTcb5B0vjwSC6QJnwJlLRBJQlVcM2WDl/view?usp=sharing) |

\* We adopt a one-to-one assignment in POTO and a one-to-many assignment in the auxiliary loss, respectively.

- It's normal to observe ~0.3AP noise in POTO, and ~1.0mMR noise in all methods.

## Ablations on COCO2017 val set

| model | assignment | with NMS | lr sched. | mAP | mAR | note |
|:------|:----------:|:--------:|:---------:|:---:|:---:|:----:|
| [POTO](./playground/detection/coco/poto.res50.fpn.coco.800size.6x_ms) | one-to-one | No | 6x + ms | 40.0 | 61.9 | |
| [POTO](./playground/detection/coco/poto.res50.fpn.coco.800size.9x_ms) | one-to-one | No | 9x + ms | 40.2 | 62.3 | |
| [POTO](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.argmax) | one-to-one | No | 3x + ms | 39.2 | 61.1 | replace Hungarian algorithm by `argmax` |
| [POTO + 3DMF](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf_wo_gn) | one-to-one | No | 3x + ms | 40.9 | 62.0 | remove GN in 3DMF |
| [POTO + 3DMF + Aux](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf_wo_gn.aux) | mixture\* | No | 3x + ms | 41.5 | 61.5 | remove GN in 3DMF |

\* We adopt a one-to-one assignment in POTO and a one-to-many assignment in the auxiliary loss, respectively.

- For `one-to-one` assignment, more training iters lead to higher performance.
- The `argmax` (also known as top-1) operation is indeed the approximate solution of bipartite matching in dense prediction methods.
- It seems harmless to remove GN in 3DMF, which also leads to higher inference speed.

## Acknowledgement
This repo is developed based on cvpods. Please check [cvpods](https://github.com/Megvii-BaseDetection/cvpods) for more details and features.

## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.

## Citing
If you use this work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:
```
@article{wang2020end,
  title   =  {End-to-End Object Detection with Fully Convolutional Network},
  author  =  {Wang, Jianfeng and Song, Lin and Li, Zeming and Sun, Hongbin and Sun, Jian and Zheng, Nanning},
  journal =  {arXiv preprint arXiv:2012.03544},
  year    =  {2020}
}
```

## Contributing to the project
Any pull requests or issues about the implementation are welcome. If you have any issue about the library (e.g. installation, environments), please refer to [cvpods](https://github.com/Megvii-BaseDetection/cvpods).
