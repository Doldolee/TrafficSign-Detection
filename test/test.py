from mmdet.apis import init_detector, inference_detector
import mmcv
import copy
import os.path as osp
import cv2

import mmcv
import numpy as np

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.apis import single_gpu_test
from mmcv.parallel import MMDataParallel
# from ..train import config

def main(name, ckpt):
    @DATASETS.register_module(force=True)
    class TSDataset(CustomDataset):
        CLASSES=('pn','pne','i5','p11','pl40','pl50','pl80','pl60','p26','i4','pl100','pl30','pl5','il60'\
            ,'i2','i2r','p5','w57','p10','p13','ip','i4l','pl120','il80','p23','pr40','w59','ph4.5','p12'\
                ,'w55','p3','pl20','pm20','pg','pl70','pm55','p27','il100','p19','w13','ph5','ph4','p6','w32','pm30')

        def load_annotations(self, ann_file):
            print('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:', self.img_prefix)
            print('#### ann_file:', ann_file)
            
            cat2label = {k:i for i,k in enumerate(self.CLASSES)}
            image_list = mmcv.list_from_file(self.ann_file)
            
            data_infos = []
            
            for image_id in image_list:
                filename = '{0:}/{1:}.jpg'.format(self.img_prefix, image_id)
                image = cv2.imread(filename)
                height, width = image.shape[:2]
                
                data_info = {'filename': str(image_id) + '.jpg','width': width, 'height': height}
                label_prefix = self.img_prefix.replace('data', 'label')
                lines = mmcv.list_from_file(osp.join(label_prefix, str(image_id)+'.txt'))
  
                content = [line.strip().split(' ') for line in lines] 
                bbox_names = [x[0] for x in content]
                bboxes = [ [float(info) for info in x[1:5]] for x in content]

                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []

                for bbox_name, bbox in zip(bbox_names, bboxes):
                    if bbox_name in cat2label:
                        gt_bboxes.append(bbox)
                        gt_labels.append(cat2label[bbox_name])
                    else:
                        gt_bboxes_ignore.append(bbox)
                        gt_labels_ignore.append(-1)
                data_anno = {
                    'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    'labels': np.array(gt_labels, dtype=np.compat.long),
                    'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                    'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
                    }
                data_info.update(ann=data_anno)
                data_infos.append(data_info)

            return data_infos

    if name =="yolox":
        cfg = config.yolox(test=True)
    elif name=="RetinaNet":
        cfg = config.RetinaNet(test=True)
    elif name=="RCNN":
        cfg = config.RCNN(test=True)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    checkpoint_file = f'../tutorial_exps/epoch_50_{ckpt}.pth'

    model_ckpt = init_detector(cfg, checkpoint_file, device='cuda:0')
    model_ckpt = MMDataParallel(model_ckpt, device_ids=[0])

    outputs = single_gpu_test(model_ckpt, data_loader, True, '../show_test_output', 0.6)
    metric = dataset.evaluate(outputs, metric='mAP')
    print(metric)


if __name__ =="__main__":
    if __package__ is None:
        import sys
        from os import path
        print(path.dirname( path.dirname( path.abspath(__file__) ) ))
        sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
        from train import config
    else:
        from ..train import config
    main(name='RCNN',ckpt='RCNN')

