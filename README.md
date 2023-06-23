# YOLOv5-for-Remote-Sensing-Object-Counting
This repository contains the code for remote sensing object counting using the YOLO algorithm, which uses YOLOv5 as the pre-trained weight.

![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/detect_result.png)


## Dataset 
Download RSOC_small-vehicle, RSOC_large-vehicle and RSOC_ship datasets from [here](https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method). These datasets are collected from the [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html), which is a very large dataset built for object detection in aerial images. 


## Preprocess
### 1) Image Split
In response to the large size of the satellite images, we divided the original images into sub-images based on the object distribution positions and adjusted the size of each sub-image to 1024 × 1024. 

![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/show_image/image_split.png)

### 2) Label Transform
The label format in the original dataset is as follows.

```
<x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <class> <difficulty>
``` 

* `x(1-4)` and `y(1-4)` are four coordinate points of the object box. 
* `class` is the object category. 
* `difficulty` is the detection difficulty (0/1: simple/difficult).

We provide `preprocess->label_transform.py` to convert this label format to the specific format for YOLO model. 

```
<class> <x_center> <y_center> <width> <height>
``` 

* `class` is the object category. 
* `x_center` is the ratio of the coordinate x of the center point of the object box to the width of the image. 
* `y_center` is the ratio of the coordinate y of the center point of the object box to the height of the image. 
* `width` is the ratio of the width of the object box to the width of the image. 
* `height` is the ratio of the height of the object box to the height of the image. 


## ONNX Runtime Deploy
We transformed the pytorch model into ONNX format and used the Microsoft inference framework ONNX Runtime to perform inference detection on the video taken by drones. The following are two examples of detecting and counting small-vehicle and ship, respectively.
![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/small-vehicle.gif)
![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/ship.gif)

## Visualization
The following are some detection examples from the RSOC_small-vehicle dataset. Detection-based methods do not perform well in highly congested scenarios, as can be seen from the red rectangular box in the right image. 

![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/show_image/result.png)


## Environment

Configuration: 
* GPU: NVIDIA Titan X
* Memory: 12GB
* CPU: Intel Xeon E5-2640Wv4 
* Operating system: Ubuntu 16.04.1

Runtime: 
* python: 3.7
* pytorch: 1.4.0  
* torchvision: 0.5.0
* cuda: 9.2 
* numpy: 1.19.4


## Code Structure

* `preprocess`: code for splitting images and transforming labels. 
* `train`: code for training. 
* `test`: code for evaluation. 
* `detect`: code for detecting objects in remote sensing images. 
* `object_count`: code for counting objects. 
