
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from PIL import Image
import cv2
from torchsummary import summary
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

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
        for i, instance in enumerate(train_0_anns, start=1):
            segmentation = instance['segmentation']
            vertices = np.array(segmentation).reshape(-1, 2)
            ImageDraw.Draw(mask).polygon(xy=vertices.ravel().tolist(), outline=1, fill=1)
            
        mask = np.array(mask)
        # Save the mask to a file
        image_path = os.path.join(output_dir,f"mask_{file_name}")
        mask_save_path = os.path.join(image_path)
        mask_image = Image.fromarray(obj=mask.astype(np.uint8))
        mask_image.save(mask_save_path)