import data_utils as util
import os
import numpy as np
from PIL import Image


# trans cur label format to format YOLO required
def labelTransfrom(imgpath, txtpath, dstpath, extractclassname):
    """
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    :return:
    """
    filelist = util.GetFileFromThisRootDir(txtpath)
    for fullname in filelist:
        objects = util.parse_dota_poly(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]
        img_fullname = os.path.join(imgpath, name + '.png')
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        # print img_w,img_h
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']
                bbox = np.array(util.dots4ToRecC(poly, img_w, img_h))
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])
                else:
                    continue
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))
                f_out.write(outline + '\n')


if __name__ == '__main__':
    # filedir = os.path.dirname(os.path.abspath(__file__))
    filedir = "/media/ysliu/6b94d4ca-f5c4-46ae-8497-af46d2544dfc/Maoer/object detection/Yolov5s"
    labelTransfrom(os.path.join(filedir, 'RSOC_small-vehicle', 'train_split', 'images'),
                   os.path.join(filedir, 'RSOC_small-vehicle', 'train_split', 'labelTxt'),
                   os.path.join(filedir, 'RSOC_small-vehicle', 'train_split', 'labels'),
                   util.wordname_15)
    labelTransfrom(os.path.join(filedir, 'RSOC_small-vehicle',  'val_split', 'images'),
                   os.path.join(filedir, 'RSOC_small-vehicle',  'val_split', 'labelTxt'),
                   os.path.join(filedir, 'RSOC_small-vehicle',  'val_split', 'labels'),
                   util.wordname_15)
