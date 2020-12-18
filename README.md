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
| [FCOS](./playground/detection/coco/fcos.res50.fpn.coco.800size.3x_ms) | one-to-many | Yes | 3x + ms | 41.1 | 59.0 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/Ed5kFlrTzaRCuTtNrEpEyvcBvE3lmwv7fhN3WlBKUQN9IQ?e=LyXyQ8) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EeHivd6wYi9NnU46O85c5xEBRjKa9T7Ao1A6UTPTk78tAQ?e=Kh9pAA) |
| [FCOS baseline](./playground/detection/coco/fcos.res50.fpn.coco.800size.3x_ms.wo_ctrness) | one-to-many | Yes | 3x + ms | 40.7 | 58.1 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EWiLRCqKVWZHvn3kjhx7aCsB8CIrecsK7K5VuVgTQVaonA?e=TW79W2) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EQPmXtL8XVxMmWbO7ikqF28BdXDckIsmBTF77cWIGxuCoA?e=vtScEJ) |
| [Anchor](./playground/detection/coco/anchor.res50.fpn.coco.800size.3x_ms) | one-to-one | No | 3x + ms | 37.1 | 60.1 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EfQV5I0pE2lEuMAQYIcN3MUBtOKWJhOSV3Fkv9Qx7hYfqA?e=M7jM2a) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EVJi5DgrRHtPq4BUB3EA0NQBvdHFGgk_lgrcE2I8l6Gf1w?e=ADHOhn) |
| [Center](./playground/detection/coco/center.res50.fpn.coco.800size.3x_ms) | one-to-one | No | 3x + ms | 34.9 | 60.7 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EY7ubzFIKHlMm9X_B3EnOmwB_nVoS-ppscXrSnLNqACGww?e=gniPNQ) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EYiKdUPKGShPnZbLKf_oRpwBiX2mH9DBCwdjjqvWIxPB2w?e=rTxDI0) |
| [Foreground Loss](./playground/detection/coco/loss.res50.fpn.coco.800size.3x_ms) | one-to-one | No | 3x + ms | 38.5 | 62.2 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EUNLYB_qIRZKlLZ8UAGgO3YBmKtuVrsNA3CVLVFk_NKvKw?e=mbT2kY) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EV43J3LWqVFNqWKKVBb6GOEBjJO7uHj7i1HVWNnAZBu1_g?e=YKfJs9) |
| [POTO](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms) | one-to-one | No | 3x + ms | 39.0 | 61.6 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/ETyzm_Tdl91EiD2JuXP_WTkByMN_peE6hhPTezlpWT4-FQ?e=a3MstA) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/Ebinr1lWsuRAnISaR4giC6gBVZ7hEoM5A992QbHsiTJcVg?e=tIR03C) |
| [POTO + 3DMF](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf) | one-to-one | No | 3x + ms | 40.3 | 61.4 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EdyQqSlekf9Avpc4DrHokvABQOVt9T29ISvkUSKlIPkbcA?e=df6D2Y) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EbLZyKkQazNAl-_wjqRSyiMB3g2kygx9HshgL3-el_7wEg?e=6JXyxf) |
| [POTO + 3DMF + Aux](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf.aux) | mixture\* | No | 3x + ms | 41.5 | 61.4 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EYgGs9PXLDVBsRxD7fluh7YBnAndyoOi7KzEdqMkB0vFZg?e=olYUaQ) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EaeB2tlXYmVBoKKtVEtGUOMBuLZNOUQtQ3iTksPdPCFAJw?e=ZwmlDB) |

\* We adopt a one-to-one assignment in POTO and a one-to-many assignment in the auxiliary loss, respectively.

- `2x + ms` schedule is adopted in the paper, but we adopt `3x + ms` schedule here to achieve higher performance.
- It's normal to observe ~0.3AP noise in POTO.

## Results on CrowdHuman val set

| model | assignment | with NMS | lr sched. | AP50 | mMR | recall | download |
|:------|:----------:|:--------:|:---------:|:----:|:---:|:------:|:--------:|
| [FCOS](./playground/detection/crowdhuman/fcos.res50.fpn.crowdhuman.800size.30k) | one-to-many | Yes | 30k iters | 86.0 | 55.2 | 94.1 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EYDm7cRXaNhKsaQIfF8a3okB2shsPxORtQsA8hlmUI9NjQ?e=eIcsVa) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/ESTXovoQaRZHp3XayoZ8XgwB1FUvxnPvUzgO-oqZfoKsXg?e=EWGWUW) |
| [ATSS](./playground/detection/crowdhuman/atss.res50.fpn.crowdhuman.800size.30k) | one-to-many | Yes | 30k iters | 87.1 | 50.3 | 94.0 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EYHbdIkq4eRLhJoytZWiipwBC9JYzfdWPl3CFCovEMuRBg?e=25LKYw) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/Eb5qjd82AA9PgVNJfnUOK9oBefCamm3qLqMNKTR0VVCETg?e=1JbYab) |
| [POTO](./playground/detection/crowdhuman/poto.res50.fpn.crowdhuman.800size.30k) | one-to-one | No | 30k iters | 88.7 | 52.0 | 96.5 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EXZ6XWt7xghIjH2ZoF5srzgBgzunrF18KmDFjDJX5XJTVg?e=hO0a7b) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EVHXlKqh4R1Fr20pvds0gYYB7uTfRyln623HtThNUeuhuA?e=Cv7dIQ) |
| [POTO + 3DMF](./playground/detection/crowdhuman/poto.res50.fpn.crowdhuman.800size.30k.3dmf) | one-to-one | No | 30k iters | 89.0 | 51.3 | 96.9 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EefGi-DNV-tNvjhfCLNllqEBa0uib_ZDwZ1jPPb-gW-IzQ?e=0EfCLB) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/ER4DWUc_FcZKudgficcvc6kBWcmIW3OB4eTLEqq2OkUvFQ?e=1QsyZK) |
| [POTO + 3DMF + Aux](./playground/detection/crowdhuman/poto.res50.fpn.crowdhuman.800size.30k.3dmf.aux) | mixture\* | No | 30k iters | 89.3 | 48.9 | 96.6 | [weight](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/EZSc1mSB495KlDcokhnHGLgBfbUMMZdBOJWPLr4AVrS9_w?e=QAQzd9) \| [log](https://megvii-my.sharepoint.cn/:u:/g/personal/wangjianfeng_megvii_com/Eei8vxwWwq9Ak8P2kUQ3WsEB7-N2dQBk6U_ck7FQKOKnzA?e=0YSC6v) |

\* We adopt a one-to-one assignment in POTO and a one-to-many assignment in the auxiliary loss, respectively.

- It's normal to observe ~0.3AP noise in POTO, and ~1.0mMR noise in all methods.

## Ablations on COCO2017 val set

| model | assignment | with NMS | lr sched. | mAP | mAR | note |
|:------|:----------:|:--------:|:---------:|:---:|:---:|:----:|
| [POTO](./playground/detection/coco/poto.res50.fpn.coco.800size.6x_ms) | one-to-one | No | 6x + ms | 39.8 | 62.1 | |
| [POTO](./playground/detection/coco/poto.res50.fpn.coco.800size.9x_ms) | one-to-one | No | 9x + ms | 40.2 | 62.2 | |
| [POTO](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.argmax) | one-to-one | No | 3x + ms | 39.0 | 61.3 | replace Hungarian algorithm by `argmax` |
| [POTO + 3DMF](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf_wo_gn) | one-to-one | No | 3x + ms | 40.7 | 61.9 | remove GN in 3DMF |
| [POTO + 3DMF + Aux](./playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf_wo_gn.aux) | mixture\* | No | 3x + ms | 41.6 | 61.5 | remove GN in 3DMF |

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
