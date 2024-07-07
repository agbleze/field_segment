# -*- coding: utf-8 -*-
"""Detectron2 for Field Area Segmentation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sNOg6sQ43bKSnyAt33WsMWiDH1gGX5xH

# MASK-RCNN for Field Area Segmentation

Colab Link: https://colab.research.google.com/drive/1sNOg6sQ43bKSnyAt33WsMWiDH1gGX5xH?usp=sharing

In this Topic, we'll explore how to leverage MASK-RCNN Mask Segmentation for the Field Area Segmentation competition. The MASK-RCNN model is an extension of RCNN, initially designed for object detection. However, MASK RCNN goes beyond simple bounding box detection by also generating masks for detected objects within an image.

Given that each image in this competition may contain thousands of masks, traditional Segmentation models like Unet and its variants may not perform optimally. These models typically generate a single mask for each image, which can lead to information loss or incorrect segmentation when breaking the mask into smaller segments for individual fields. Therefore, I believe that Mask-RCNN could offer a promising solution to address this challenge. Since it can simultaneously generate thousands of masks of field in a single image, eliminating the need to manually partition the mask into smaller components.

In my submission, I achieved scores of **0.0026 and 0.0021** for different thresholds. However, please note that there is a degree of randomness involved, so your results may vary. This serves as a simple baseline, and there are several avenues for improvement, including:

1. Incorporating augmentations to enhance model performance.
2. Experimenting with different base models.
3. Adjusting hyperparameters to optimize model performance.
4. Exploring various methods for utilizing and combining the 12 channels in the TIFF image data.
5. Do cross validations (I am skipping validation in this notebook)

There's ample room for experimentation and improvement with this approach, and I hope this post serves as a helpful starting point for your exploration!

Also note that this is an adaptation of the orginal detectron2 tutorial notebook, and detectron2 is under Apache-2.0 license, so I think we can use it.

# Install detectron2
"""
#%%
!python -m pip install pyyaml==5.1 -q
import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])} -q
sys.path.insert(0, os.path.abspath('./detectron2'))

# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#%%
import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
#%% Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

#%% import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

#%% import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

"""# Import All the relevent libaries"""
#%%
import os, gc, sys, time, random, copy, json
from IPython.display import display
#%%
if not os.path.exists('drive'):
    from google.colab import drive
    drive.mount('/content/drive')

#%%
import torch
from torchvision import transforms, ops, models

try:
    import rasterio
except:
    !pip install rasterio -q

import rasterio
from rasterio.plot import show

from tqdm.notebook import tqdm
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import albumentations as A

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from shapely.geometry import Polygon

print(os.cpu_count())
#%%
"""# Train on a custom dataset

In this section, we will initially reduce the 12-channel TIFF image into a 3-channel PNG image suitable for use with Mask-RCNN. This process involves selecting the RGB channel and normalizing the images to the range of [0, 1].

#### Note on Data Storage Locations
**Please ensure that your training and testing data are stored in the 'DATA/train_images_satellite' and 'DATA/test_images_satellite' folders, respectively. Annotations should be stored in the 'DATA/train_annotation.json' file.**

The image data will then be converted and saved in the 'DATA/train_images' and 'DATA/test_images' directories.

## Dataset Preparation
"""
#%%
import os, rasterio
from PIL import Image

def generate_image(mainpath, filename, output_path):
    image_data = rasterio.open(os.path.join(mainpath, f'{filename}.tif')).read()
    image_data = np.nan_to_num(image_data, nan = 100)
    for i in range(3):
        image_data[i] = (image_data[i] - image_data[i].min()) / (image_data[i].mean() * 2 - image_data[i].min())
    image_data = image_data[1:4, :, :]
    image_data = image_data.clip(0, 1)
    image_data = (image_data * 255).astype('uint8')
    image = Image.fromarray(image_data.transpose(1, 2, 0))
    image.save(os.path.join(output_path, f'{filename}.png'))

#%%
train_mainpath = "/home/lin/codebase/field_segment/data/train_images/images"
for i in tqdm(range(50)):
    generate_image(mainpath=train_mainpath, f'train_{i}', 'DATA/train_images')
    generate_image('DATA/test_images_satellite', f'test_{i}', 'DATA/test_images')

#%%
"""# Creating the DataLoader for Detectron2

In this section, we'll define a function to load our data in a format that can be used by Detectron2.
"""

from detectron2.structures import BoxMode

def get_field_dicts(img_dir = 'DATA/train_images'):
    with open('DATA/train_annotation.json') as f:
        imgs_anns = json.load(f)['images']
    dataset_dicts = []
    for idx, v in tqdm(enumerate(imgs_anns)):
        record = {}
        filename = os.path.join(img_dir, v['file_name'].replace('.tif', '.png'))

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []
        for anno in v['annotations']:
            anno = anno['segmentation']

            obj = {
                "bbox": Polygon(np.array([[x, y] for x, y in zip(anno[::2], anno[1::2])])).bounds,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [anno],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

DatasetCatalog.register("feilds_train5", get_field_dicts)
MetadataCatalog.get("feilds_train5").set(thing_classes=["fields"])

fields_metadata = MetadataCatalog.get("feilds_train5")
fields_metadata

"""Here we can visualize the labels."""

dataset_dicts = get_field_dicts()
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fields_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])

"""## Training!

Next, let's fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on our custom dataset. You can experiment with other configurations from the [model zoo](https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py).
"""

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("feilds_train5",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = os.cpu_count()
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 1e-3     # pick a good LR
cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.GAMMA = 0.2
cfg.SOLVER.STEPS = (500, 1000, 1500)
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128, 256]]


cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  #

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""## Performing Inference with the Trained Model

In this section, we'll load the trained model for inference. I've set a confidence threshold of 0.4 for the output masks and a limit of 2000 masks per image.
"""

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE  = 2000
predictor = DefaultPredictor(cfg)

"""Now, we'll generate the prediction for submission here."""

from detectron2.utils.visualizer import ColorMode

output_file = {'images': []}
for i in tqdm(range(50)):
    file_pred = {"file_name": f"test_{i}.tif", "annotations":[]}

    im = cv2.imread(os.path.join('DATA/test_images', f'test_{i}.png'))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    result  = outputs["instances"].to("cpu")

    for i in range(len(result)):
        m = np.squeeze(result[i]._fields['pred_masks'].numpy()) * 1

        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        predicted_polygon = [ np.squeeze(contour, axis=1) for contour in contours if contour.shape[0] >= 3]
        predicted_polygon = [ po for po in predicted_polygon if Polygon(po).area >= 100]
        predicted_polygon = [ np.concatenate(po).tolist() for po in predicted_polygon ]
        for po in predicted_polygon:
            file_pred["annotations"].append({
                "class": "field","segmentation": po
            })


    output_file['images'].append(file_pred)

"""The result is dumpted into a json file for submission."""

with open('result0.4.json', 'w') as fp:
    json.dump(output_file, fp)

"""In this guide, we've walked through the process of utilizing MASK-RCNN for field area segmentation.

**If you found this guide helpful, consider upvoting it**.

If you have further questions, feel free to ask.

"""
# %%
