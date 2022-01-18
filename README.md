# Traffic Sign Detection


The Traffic Sign Detection Project is the first step to making a robust model that can apply to similar domains. 
At this project, train some detection models at TT-100k dataset and test at GTSD dataset


# Installation
- mmdetection [[start guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)]

# Data set
- [TT-100k](https://cg.cs.tsinghua.edu.cn/traffic-sign/) [5]
- [GTSD](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html) [6]

# Folder structure
```bash
├── Data
│   ├── anno_txt
│   ├── anno_xml
│   ├── Images_GTSD
│   │    ├── data
│   │    └── label
│   └── Images_TT
├── test
├── train
├── tt100k_2021
├── tutorial_exps
└── util
``` 
- Move all the TT-100k images(train,test) to the 'Images_TT' folder.
  - maybe [35542,62778,78585,79029,88586,90422].jpg were overlapped. check and remove 
- Move all the GTSD images to the 'data' folder.


# How to Run

**Create annatation files(txt, xml, json format)**
```
python check_annoJson.py

./voc2coco.sh
```

**Model Train**
```
python train_RCNN.py
```

**Model Test*
```
python test.py
```


# Models
- Faster-RCNN [1]
- RetinaNet [2]
- YoloX [3]


# Result
## Model Architecture & Performance
										
| TT-100k           | mAP      | small   | medium  | large     | small(R) | medium(R)| large(R)| 
| ----------------- | -------- | ------- | ------- | --------- | -------- | -------- | ------- | 
| Faster-RCNN       | 0.538    | 0.119   | 0.551   | 0.709     | 0.158    | 0.748    | 0.805   |
| RetinaNet         | 0.626    | 0.252   | 0.541   | 0.755     | 0.497    | 0.791    | 0.881   |
| YoloX             | 0.667    | 0.346   | 0.542   | 0.733     | 0.472    | 0.682    | 0.862   | 

## Test at GTSD [[result](https://github.com/ai-healthcare-lab/TrafficSign/blob/main/result/result.png)]
| GTSD              | mAP      |
| ----------------- | -------- |
| Faster-RCNN       | 0.500    |
| RetinaNet         | 0.446    |
| YoloX             | 0.467    |

# Reference
- [1] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (2015) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks [paper](https://arxiv.org/abs/1506.01497)
- [2] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár (2017) Focal Loss for Dense Object Detection [paper](https://arxiv.org/abs/1708.02002)
- [3] Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun (2021) YOLOX: Exceeding YOLO Series in 2021 [paper](https://arxiv.org/abs/2107.08430)
- [4] Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua (2019) MMDetection:Open MMLab Detection Toolbox and Benchmark [paper](https://arxiv.org/abs/1906.07155)
- [5] Sebastian Houben and Johannes Stallkamp and Jan Salmen and Marc Schlipsing and Christian Igel (2013) Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark
- [6] Zhe Zhu; Dun Liang; Songhai Zhang; Xiaolei Huang; Baoli Li; Shimin Hu (2016) Traffic-Sign Detection and Classification in the Wild [paper](https://ieeexplore.ieee.org/abstract/document/7780601)
- [7] https://github.com/yukkyo/voc2coco
