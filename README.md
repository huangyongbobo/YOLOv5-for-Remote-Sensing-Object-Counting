# YOLOv5-for-Remote-Sensing-Object-Counting
This repository contains the code for remote sensing object counting using the YOLO algorithm, which uses YOLOv5x as the pre-trained weight.


## Dataset 
Download RSOC_small-vehicle datasets from [here](https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method). This dataset is collected from the [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html), which is a very large dataset built for object detection in aerial images. The following are some example images from the RSOC_small-vehicle dataset. 

![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/show_image/example.png)


## Preprocess
1) Image Split


![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/show_image/image_split.png)

2) Label Transform
The label formats in the original dataset is as follows, wherr 'x(1-4)' and 'y(1-4)' are four coordinate points of the object box, 'class' is the object category, 'difficulty' is the detection difficulty (0/1: simple/difficult).
  ```
  <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <class> <difficulty>
  ```
we provide 'preprocess->label_transform.py' to convert this label formats to the specific formats for YOLO model.


## Visualization


![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/show_image/result.png)

## Environment

## Code Structure
