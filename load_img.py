
#%%
import json
from PIL import Image
import cv2
import torch
import tensorboard
import os
import tempfile
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.datamodules import EuroSAT100DataModule
from torchgeo.trainers import ClassificationTask
from torchgeo.models import ResNet18_Weights, ViTSmall16_Weights
import timm
from torchsummary import summary
import rasterio
import matplotlib.pyplot as plt
import numpy as np
# %%
train_annot_path = "/home/lin/codebase/field_segment/train_annotation.json"

#%%
with open(train_annot_path, "r") as file:
    data = json.load(file)

#%%
data["images"]
#%%
data.keys()

#%%
from cucim import CuImage

#%%
train_0_path = "/home/lin/codebase/field_segment/data/train_images/images/train_0.tif"
img = CuImage(train_0_path)

#%%
img.channel_names


#%%
cv2.imread(train_0_path)

#%%
Image.open(train_0_path)

# %%
#%%
def get_tiff_img(path, return_all_bands, bands=("B01", "B03", "B02"),
                 normalize_bands=True
                ):
    all_band_names = ("B01","B02", "B03","B04","B05", "B06",
                      "B07","B08","B8A","B09","B11","B12"
                    )
    if return_all_bands:
        band_indexs = [all_band_names.index(band_name) for band_name in all_band_names]
    
    else:
        band_indexs = [all_band_names.index(band_name) for band_name in bands]
    print(band_indexs)
    with rasterio.open(path) as src:
        img_bands = [src.read(band) for band in range(1,13)]
    dstacked_bands = np.dstack([img_bands[band_index] for band_index in band_indexs])
    #dstacked_bands = np.dstack([img_bands[3], img_bands[2], img_bands[1]])
    if normalize_bands:
        # Normalize bands to 0-255
        dstacked_bands = ((dstacked_bands - dstacked_bands.min()) / 
                          (dstacked_bands.max() - dstacked_bands.min()) * 255
                          ).astype(np.uint8)

    return dstacked_bands







#%%
default_rgb_bands = get_tiff_img(path=train_0_path, return_all_bands=False,
                                 bands=("B04","B03","B02")
                                 )

#%%
plt.imshow(default_rgb_bands)
plt.show()

# %%
default_rgb_bands.shape
# %%
def draw_masks_fromDict(image, masks_generated) :
  masked_image = image.copy()
  for i in range(len(masks_generated)) :
    masked_image = np.where(np.repeat(np.int32(masks_generated[i]['segmentation'])[:, :, np.newaxis], 3, axis=2),
                            np.random.choice(range(256), size=3),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

  return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


#%%
annot_images = data["images"]

#%%
#for ann in annot_images:
train_0_anns = [ann for ann in annot_images if ann["file_name"]=="train_0.tif"][0]["annotations"]


# %%
draw_masks_fromDict(image=default_rgb_bands, masks_generated=train_0_anns)
# %%
mask = [0.0,
   19.602,
   33.21799999999998,
   25.444999999999997,
   30.714000000000002,
   61.42,
   0.0,
   58.749]
# %%
np.int32(mask)
# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Assuming img is your image
# img = cv2.imread('your_image_path.jpg')

# Your segmentation data
data = [
    {'class': 'field', 'segmentation': [0.0, 19.602, 33.21799999999998, 25.444999999999997, 30.714000000000002, 61.42, 0.0, 58.749]},
    {'class': 'field', 'segmentation': [0.0, 58.849166512472635, 34.636999999999986, 62.254, 29.97700000000001, 97.20699999999998, 4.601999999999994, 94.069, 0.16500000000000545, 92.036]}
]

# Create an empty mask
mask = np.zeros_like(default_rgb_bands)

# Draw the polygons on the mask
r,g,b = 100, 175, 12
for obj in train_0_anns:
    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    pts = np.array(obj['segmentation']).reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, [pts], color)

# Combine the image with the mask
#img_masked = cv2.bitwise_and(default_rgb_bands, mask)
img_masked = cv2.addWeighted(default_rgb_bands, 0.7, mask, 0.1, 0)
cv2.putText(image, category_name, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
#%% Display the image
#plt.imshow(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))
plt.imshow(img_masked)
plt.show()


#%% 

def visualize_segmask(annotation_path, img_dir):
    with open(annotation_path,"r") as file:
        annot_data = json.load(file)
        
    img_data = annot_data["images"]
    for annot in img_data:
        file_name = annot["file_name"]
        train_0_anns = annot["annotations"]
        img_path = os.path.join(img_dir, file_name)
        img = get_tiff_img(img_path, return_all_bands=False, bands=("B04", "B03", "B02"))
        mask = np.zeros_like(img)
        for obj in train_0_anns:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            pts = np.array(obj['segmentation']).reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], color)
            centroid = np.mean(pts, axis=0)
            # cv2.putText(mask, f"{obj['class']}", (int(centroid[0][0]), int(centroid[0][1])), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            #             )

        # Combine the image with the mask
        #img_masked = cv2.bitwise_and(default_rgb_bands, mask)
        img_masked = cv2.addWeighted(img, 0.7, mask, 0.1, 0)
        plt.imshow(img_masked)
        plt.show()


#%%
img_dir = "/home/lin/codebase/field_segment/data/train_images/images"
visualize_segmask(annotation_path=train_annot_path, img_dir=img_dir)

        
    
# %%
print(type(default_rgb_bands))  # Should print <class 'numpy.ndarray'>
print(img.dtype)  # Should print uint8

#%%
img_masked = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
#%%
# If the type is not as expected, convert the image
if img.dtype != 'uint8':
    img = img.astype('uint8')
