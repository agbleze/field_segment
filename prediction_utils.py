

import numpy as np
import cv2
import torch
import json
import matplotlib.pyplot as plt
import os
import torch
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from glob import glob
from field_segment_utils import get_tiff_img
from PIL import Image
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
import utils
from model_obj import get_transform


from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def mask_to_polygon(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Approximate the contour to a polygon and flatten the polygon array
    polygons = [cv2.approxPolyDP(contour, 1, True).flatten().tolist() for contour in contours]
    
    return polygons

def predict_segmask(model, image, eval_transform=get_transform(train=False),
                    device="cuda", pixel_class_proba_threshold=0.5,
                    visualize=True, mask_transparency=0.3
                    ):
    image = tv_tensors.Image(image).permute(2,0,1)
    model.eval()
    with torch.no_grad():
        if eval_transform:
            x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]
    masks = (pred["masks"] > pixel_class_proba_threshold).squeeze(1)
    
    if visualize:
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]
        output_image = draw_segmentation_masks(image, masks, alpha=mask_transparency)
        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0))
    
    return masks    

#%%  
import json    
def create_prediction_annotation(model, test_img_dir,
                                 img_ext="tif",
                                save_as="predicted_segmask.json",
                                eval_transform=get_transform(train=False),
                                device="cuda", pixel_class_proba_threshold=0.5,
                                visualize=True,
                                remove_segmnet_less_four_values=True,
                                mask_transparency=0.3
                                ):
    test_img_paths = sorted(glob(f"{test_img_dir}/*.{img_ext}"))
    
    images_annot = []
    for img_path in test_img_paths:
        image_name = os.path.basename(img_path)
        image = get_tiff_img(path=img_path,
                            return_all_bands=False,
                            bands=("B04", "B03", "B02")
                            )
        masks = predict_segmask(model=model, image=image, eval_transform=eval_transform,
                                device=device, pixel_class_proba_threshold=pixel_class_proba_threshold,
                                visualize=visualize, 
                                mask_transparency=mask_transparency
                                )
        annotations = []
        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy().astype(np.uint8)
            polygons = mask_to_polygon(mask)
            for polygon in polygons:
                if remove_segmnet_less_four_values:
                    if not len(polygon) <= 4:
                        annotations.append({'class': "field", 'segmentation': polygon})
                    else:
                        print(f"{image_name} has less than 4 polygons predicted \n polgon: {polygon} \n")
                else:
                    annotations.append({'class': "field", 'segmentation': polygon})

        img_pred_annt = {"file_name": image_name, "annotations": annotations}
        images_annot.append(img_pred_annt)
    final_annot = {"images": images_annot}
    with open(save_as, "w") as file:
        json.dump(final_annot, file)   
        
    return final_annot
        