# YOLOv5-for-Remote-Sensing-Object-Counting
This repository contains the code for remote sensing object counting using the YOLO algorithm, which uses YOLOv5x as the pre-trained weight.


## Dataset 
Download RSOC_small-vehicle datasets from [here](https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method). This dataset is collected from the [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html), which is a very large dataset built for object detection in aerial images. The following are some example images from the RSOC_small-vehicle dataset. 

![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/show_image/example.png)


## Preprocess
### 1) Image Split


![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/show_image/image_split.png)

### 2) Label Transform
The label formats in the original dataset is as follows.

```
<x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <class> <difficulty>
```

`x(1-4)` and `y(1-4)` are four coordinate points of the object box. 
`class` is the object category. 
`difficulty` is the detection difficulty (0/1: simple/difficult).

we provide `preprocess->label_transform.py` to convert this label formats to the specific formats for YOLO model.
 
```
<class> <x_center> <y_center> <width> <height>
```

`class` is the object category. 
`x_center` is the ratio of the coordinates x of the center point of the object boxs to the width of the images. 
`y_center` is the ratio of the coordinates y of the center point of the object boxs to the height of the images. 
`width` is the ratio of the width of the object boxs to the width of the images. 
`height` is the ratio of the height of the object boxs to the height of the images. 


## Visualization


![](https://github.com/huangyongbobo/YOLOv5-for-Remote-Sensing-Object-Counting/blob/main/show_image/result.png)

## Environment

## Code Structure
