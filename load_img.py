
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
    #print(band_indexs)
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

#%%
annot_images = data["images"]

#%%
#for ann in annot_images:
train_0_anns = [ann for ann in annot_images if ann["file_name"]=="train_0.tif"][0]["annotations"]


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
#%% Display the image
#plt.imshow(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))
plt.imshow(img_masked)
plt.show()

#%%

cv2.imwrite("example_mask.png", mask)


#%%
import cv2
import numpy as np
contours = np.array([[50,50], [50,150], [150,150], [150,50]])
image = np.zeros((200,200))
cv2.fillPoly(image, pts = [contours], color =(255,255,255))
cv2.imshow("filledPolygon", image)
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
        img_masked = cv2.addWeighted(img, 0.7, mask, 0.1, 0)
        plt.imshow(img_masked)
        plt.show()


#%%
img_dir = "/home/lin/codebase/field_segment/data/train_images/images"

#%%
visualize_segmask(annotation_path=train_annot_path, img_dir=img_dir)


#%%

import numpy as np
from PIL import Image, ImageDraw

#%%
#ex_image = Image.open(train_0_path)
height, width, num_chan = default_rgb_bands.shape

# Initialize an empty mask
ex_mask = Image.new('L', (width, height), 0)

#%% Draw each instance's polygon on the mask with a unique pixel value
for i, instance in enumerate(train_0_anns, start=1):
    segmentation = instance['segmentation']
    vertices = np.array(segmentation).reshape(-1, 2)
    ImageDraw.Draw(ex_mask).polygon(xy=vertices.ravel().tolist(), outline=i, fill=i)

#%% Convert the mask to a numpy array
ex_mask = np.array(ex_mask)

# Save the mask to a file
mask_save_path = 'train_0_mask.tif'
mask_image = Image.fromarray(obj=ex_mask.astype(np.uint8))
mask_image.save(mask_save_path)

# Visualize the mask
import matplotlib.pyplot as plt
plt.imshow(X=mask, cmap='gray')
plt.show()


#%%
def create_instance_segmask(annotation_path, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(annotation_path,"r") as file:
        annot_data = json.load(file)
        
    img_data = annot_data["images"]
    for annot in img_data:
        file_name = annot["file_name"]
        train_0_anns = annot["annotations"]
        img_path = os.path.join(img_dir, file_name)
        img = get_tiff_img(img_path, return_all_bands=False, bands=("B04", "B03", "B02"))
        height, width, num_chan = img.shape
        mask = Image.new('L', (width, height), 0)
        print(file_name)
        for i, instance in enumerate(train_0_anns, start=1):
            segmentation = instance['segmentation']
            vertices = np.array(segmentation).reshape(-1, 2)
            ImageDraw.Draw(mask).polygon(xy=vertices.ravel().tolist(), outline=i, fill=i)
            
        mask = np.array(mask)
        # Save the mask to a file
        image_path = os.path.join(output_dir,f"mask_{file_name}")
        mask_save_path = os.path.join(image_path)
        mask_image = Image.fromarray(obj=mask.astype(np.uint8))
        mask_image.save(mask_save_path)
        
#%%

create_instance_segmask(annotation_path=train_annot_path, img_dir=img_dir,
                        output_dir="masks_smp"
                        )


#%%

img_37_path = "/home/lin/codebase/field_segment/data/train_images/images/train_37.tif"

get_tiff_img(path=img_37_path, return_all_bands=False, normalize_bands=True)
#%% Initialize an empty mask
ex_mask = Image.new('L', (width, height), 0)

#%% Draw each instance's polygon on the mask with a unique pixel value
for i, instance in enumerate(train_0_anns, start=1):
    segmentation = instance['segmentation']
    vertices = np.array(segmentation).reshape(-1, 2)
    ImageDraw.Draw(ex_mask).polygon(xy=vertices.ravel().tolist(), outline=i, fill=i)

#%% Convert the mask to a numpy array
ex_mask = np.array(ex_mask)

# Save the mask to a file
mask_save_path = 'train_0_mask.png'
mask_image = Image.fromarray(obj=ex_mask.astype(np.uint8))
mask_image.save(mask_save_path)

# Visualize the mask
import matplotlib.pyplot as plt
plt.imshow(X=mask, cmap='gray')
plt.show()


#%%

read_img = Image.open("/home/lin/codebase/field_segment/masks/mask_train_0.tif")

#%%
from PIL import ImageDraw
def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)
    
    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img


#%%
img_masked = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
#%%
# If the type is not as expected, convert the image
if img.dtype != 'uint8':
    img = img.astype('uint8')


