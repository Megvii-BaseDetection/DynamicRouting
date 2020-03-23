import os.path as osp

from dl_lib.configs.segm_config import SemanticSegmentationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/data/dl_lib_modelzoo/dynamicmodel/Dynamic-L33B4-A74-convert-seg.pth",
        CAL_FLOPS=True,
        BACKBONE=dict(
            CELL_TYPE=['sep_conv_3x3', 'skip_connect'],
            LAYER_NUM=33,
            CELL_NUM_LIST=[2, 3, 4] + [4 for _ in range(30)],
            INIT_CHANNEL=64,
            MAX_STRIDE=32,
            SEPT_STEM=True,
            NORM="nnSyncBN",
            DROP_PROB=0.0,
        ),
        GATE=dict(
            GATE_ON=True,
            GATE_INIT_BIAS=1.5,
            SMALL_GATE=False,
        ),
        SEM_SEG_HEAD=dict(
            IN_FEATURES=['layer_0', 'layer_1', 'layer_2', 'layer_3'],
            NUM_CLASSES=19,
            IGNORE_VALUE=255,
            NORM="nnSyncBN",
            LOSS_WEIGHT=1.0,
        ),
        BUDGET=dict(
            CONSTRAIN=False,
            LOSS_WEIGHT=0.0,
            LOSS_MU=0.0,
            FLOPS_ALL=26300.0,
            UNUPDATE_RATE=0.4,
            WARM_UP=True,
        ),
    ),
    DATASETS=dict(
        TRAIN=("cityscapes_fine_sem_seg_train", ),
        TEST=("cityscapes_fine_sem_seg_val", ),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="PolyLR",
            POLY_POWER=0.9,
            MAX_ITER=190000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
            GATE_LR_MULTI=2.5,
        ),
        IMS_PER_BATCH=8,
        CHECKPOINT_PERIOD=5000,
        GRAD_CLIP=5.0,
    ),
    INPUT=dict(
        MIN_SIZE_TRAIN=(512, 768, 1024, 1280, 1536, 2048, ),
        MIN_SIZE_TRAIN_SAMPLING="choice",
        MAX_SIZE_TRAIN=4096,
        MIN_SIZE_TEST=1024,
        MAX_SIZE_TEST=2048,
        # FIX_SIZE_FOR_FLOPS=[768,768],
        FIX_SIZE_FOR_FLOPS=[1024, 2048],
        CROP_PAD=dict(SIZE=[768, 768], ),
    ),
    TEST=dict(
        AUG=dict(
            ENABLED=False,
            MIN_SIZES=(512, 768, 1024, 1280, 1536, 2048, ),
            MAX_SIZE=4096,
            FLIP=True,
        ),
        PRECISE_BN=dict(ENABLED=True),
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/dl_lib_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class DynamicSemanticSegmentationConfig(SemanticSegmentationConfig):
    def __init__(self):
        super(DynamicSemanticSegmentationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = DynamicSemanticSegmentationConfig()
