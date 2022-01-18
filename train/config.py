from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv

def RCNN(test=False):

    config_file = "../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py"
    checkpoint_file = "../mmdetection/checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"

    cfg = Config.fromfile(config_file)

    cfg.dataset_type = 'TSDataset'
    cfg.data_root = '../Data/'

    cfg.data.train.type = 'TSDataset'
    cfg.data.train.data_root = '../Data/'
    cfg.data.train.ann_file = 'train_TT.json'
    cfg.data.train.img_prefix = 'Images_TT'

    cfg.data.val.type = 'TSDataset'
    cfg.data.val.data_root = '../Data/'
    cfg.data.val.ann_file = 'val_TT.json'
    cfg.data.val.img_prefix = 'Images_TT'

    if test:
        cfg.data.test.type = 'TSDataset'
        cfg.data.test.data_root = '../Data/'
        cfg.data.test.ann_file = 'test_ids_gtsd.txt'
        cfg.data.test.img_prefix = 'Images_GTSD/data'
    else:
        cfg.data.test.type = 'TSDataset'
        cfg.data.test.data_root = '../Data/'
        cfg.data.test.ann_file = 'val_TT.json'
        cfg.data.test.img_prefix = 'Images_TT'

    cfg.model.roi_head.bbox_head.num_classes = 45
    cfg.load_from = checkpoint_file

    cfg.work_dir = '../tutorial_exps'

    cfg.optimizer.lr = 0.04/8
    cfg.optimizer.weight_decay=0.0005
    cfg.optimizer_config.grad_clip=dict(max_norm=35, norm_type=2)
    cfg.log_config.interval = 10

    cfg.evaluation.metric = 'bbox'
    cfg.evaluation.interval = 25
    cfg.checkpoint_config.interval = 25

    cfg.lr_config.policy='step'
    cfg.data.samples_per_gpu = 1
    cfg.runner.max_epochs=50
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    return cfg

def RetinaNet(test=False):
    config_file = '../mmdetection/configs/retinanet/retinanet_r50_fpn_mstrain_640-800_3x_coco.py'
    checkpoint_file = '../mmdetection/checkpoints/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth'

    cfg = Config.fromfile(config_file)

    cfg.dataset_type = 'TSDataset'
    cfg.data_root = '../Data/'

    cfg.data.train.dataset.type = 'TSDataset'
    cfg.data.train.dataset.data_root = '../Data/'
    cfg.data.train.dataset.ann_file = 'train_TT.json'
    cfg.data.train.dataset.img_prefix = 'Images_TT'

    cfg.data.val.type = 'TSDataset'
    cfg.data.val.data_root = '../Data/'
    cfg.data.val.ann_file = 'val_TT.json'
    cfg.data.val.img_prefix = 'Images_TT'

    if test:
        cfg.data.test.type = 'TSDataset'
        cfg.data.test.data_root = '../Data/'
        cfg.data.test.ann_file = 'test_ids_gtsd.txt'
        cfg.data.test.img_prefix = 'Images_GTSD/data'
    else:
        cfg.data.test.type = 'TSDataset'
        cfg.data.test.data_root = '../Data/'
        cfg.data.test.ann_file = 'val_TT.json'
        cfg.data.test.img_prefix = 'Images_TT'

    cfg.model.bbox_head.num_classes = 45


    cfg.load_from = checkpoint_file
    cfg.work_dir = '../tutorial_exps'

    cfg.optimizer.lr = 0.04/8
    cfg.optimizer.weight_decay=0.0005
    cfg.optimizer_config.grad_clip=dict(max_norm=35, norm_type=2)
    cfg.log_config.interval = 26

    cfg.evaluation.metric = 'bbox'
    cfg.evaluation.interval = 26
    cfg.checkpoint_config.interval = 26
    cfg.data.samples_per_gpu = 1

    cfg.lr_config.policy='step'
    cfg.lr_config.warmup_iters=1000
    cfg.runner.max_epochs=52

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    return cfg

def yolox(test=False):
    config_file = '../mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py'
    checkpoint_file = '../mmdetection/checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

    cfg = Config.fromfile(config_file)

    cfg.dataset_type = 'TSDataset'
    cfg.data_root = '../Data/'

    cfg.data.train.dataset.type = 'TSDataset'
    cfg.data.train.dataset.data_root = '../Data/'
    cfg.data.train.dataset.ann_file = 'train_TT.json'
    cfg.data.train.dataset.img_prefix = 'Images_TT'

    cfg.data.val.type = 'TSDataset'
    cfg.data.val.data_root = '../Data/'
    cfg.data.val.ann_file = 'val_TT.json'
    cfg.data.val.img_prefix = 'Images_TT'

    if test:
        cfg.data.test.type = 'TSDataset'
        cfg.data.test.data_root = '../Data/'
        cfg.data.test.ann_file = 'test_ids_gtsd.txt'
        cfg.data.test.img_prefix = 'Images_GTSD/data'
    else:
        cfg.data.test.type = 'TSDataset'
        cfg.data.test.data_root = '../Data/'
        cfg.data.test.ann_file = 'val_TT.json'
        cfg.data.test.img_prefix = 'Images_TT'
        

    cfg.model.bbox_head.num_classes = 45

    cfg.load_from = checkpoint_file

    cfg.work_dir = '../tutorial_exps'

    cfg.optimizer.lr = 0.04/8
    cfg.optimizer.weight_decay=0.0005
    cfg.optimizer_config.grad_clip=dict(max_norm=35, norm_type=2)
    cfg.log_config.interval = 24

    cfg.evaluation.metric = 'bbox'
    cfg.evaluation.interval = 24
    cfg.checkpoint_config.interval = 24
    cfg.data.samples_per_gpu = 1

    cfg.runner.max_epochs=48
    cfg.max_epochs=48

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1,2)

    return cfg
