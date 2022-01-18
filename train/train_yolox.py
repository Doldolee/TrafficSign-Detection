from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp
import mmcv
import config


def main(algo):
    @DATASETS.register_module(force=True)
    class TSDataset(CocoDataset):
        CLASSES=('pn','pne','i5','p11','pl40','pl50','pl80','pl60','p26','i4','pl100','pl30','pl5','il60'\
            ,'i2','i2r','p5','w57','p10','p13','ip','i4l','pl120','il80','p23','pr40','w59','ph4.5','p12'\
                ,'w55','p3','pl20','pm20','pg','pl70','pm55','p27','il100','p19','w13','ph5','ph4','p6','w32','pm30')

    if algo == "yolox":
        cfg = config.yolox()


    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].dataset.CLASSES
    print("enrolled class")
    print(model.CLASSES)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == "__main__":
    main(algo='yolox')