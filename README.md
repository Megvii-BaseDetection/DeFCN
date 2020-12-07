# End-to-End Object Detection with Fully Convolutional Network

![GitHub](https://img.shields.io/github/license/Megvii-BaseDetection/DeFCN)

This project provides an implementation for "[End-to-End Object Detection with Fully Convolutional Network](https://arxiv.org/abs/2012.03544)" on PyTorch.

For the reason that experiments in the paper were conducted using internal framework, this project reimplements them on cvpods and reports detailed comparisons below.

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
cd DeFCN/playground/detection/coco/experiment  # for example

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

Comming soon

## Results on CrowdHuman val set

Comming soon

## Acknowledgement
This repo is developed based on cvpods. Please check [cvpods](https://github.com/Megvii-BaseDetection/cvpods) for more details and features.

## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.

## Citing
If you use this work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:
```
@article{wang2020end,
  title   =  {End-to-End Object Detection with Fully Convolutional Network},
  author  =  {Jianfeng Wang and Lin Song and Zeming Li and Hongbin Sun and Jian Sun and Nanning Zheng},
  journal =  {arXiv preprint arXiv:2012.03544},
  year    =  {2020}
}
```

## Contributing to the project
Any pull requests or issues are welcome.
