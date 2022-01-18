#! /bin/bash

git clone https://github.com/yukkyo/voc2coco.git

cd voc2coco

python voc2coco.py --ann_dir ../Data/anno_xml \
--ann_ids ../Data/train_ids_TT.txt \
--labels ../util/class.txt \
--output ../Data/train_TT.json \
--ext xml

python voc2coco.py --ann_dir ../Data/anno_xml \
--ann_ids ../Data/val_ids_TT.txt \
--labels ../util/class.txt \
--output ../Data/val_TT.json \
--ext xml
