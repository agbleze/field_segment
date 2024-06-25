
#%%
import json
from PIL import Image
import cv2
import torch

# %%
train_annot_path = "/home/lin/codebase/field_segment/train_annotation.json"

#%%
with open(train_annot_path, "r") as file:
    data = json.load(file)

#%%
data["images"]












# %%
