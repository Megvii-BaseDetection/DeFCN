import os.path as osp

from cvpods.configs.fcos_config import FCOSConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        RESNETS=dict(DEPTH=50),
        SHIFT_GENERATOR=dict(
            NUM_SHIFTS=1,
            OFFSET=0.5,
        ),
        FCOS=dict(
            NUM_CLASSES=1,
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            CENTER_SAMPLING_RADIUS=1.5,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
        ),
    ),
    DATASETS=dict(
        TRAIN=("crowdhuman_train",),
        TEST=("crowdhuman_val",),
    ),
    SOLVER=dict(
        CHECKPOINT_PERIOD=5000,
        LR_SCHEDULER=dict(
            MAX_ITER=30000,
            STEPS=(20000, 25000),
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(short_edge_length=(800,), max_size=1400, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(short_edge_length=800, max_size=1400, sample_style="choice")),
            ],
        )
    ),
    TEST=dict(
        DETECTIONS_PER_IMAGE=500,
        EVAL_PEROID=5000,
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
