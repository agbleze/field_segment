
import rasterio
import matplotlib.pyplot as plt
import numpy as np

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

