
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os, json, cv2

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
