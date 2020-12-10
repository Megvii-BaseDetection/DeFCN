import os.path as osp

from cvpods.configs.fcos_config import FCOSConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32], [64], [128], [256], [512]],
            ASPECT_RATIOS=[[1.0]],
            OFFSET=0.5,
        ),
        FCOS=dict(
            NMS_THRESH_TEST=1.0,  # disable NMS when NMS threshold is 1.0
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            REG_WEIGHT=2.0,
        ),
        POTO=dict(
            IOU_TOPK=1,
        ),
        NMS_TYPE=None,
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        CHECKPOINT_PERIOD=10000,
        LR_SCHEDULER=dict(
            MAX_ITER=270000,
            STEPS=(210000, 250000),
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    TEST=dict(
        EVAL_PEROID=10000,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class CustomFCOSConfig(FCOSConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
