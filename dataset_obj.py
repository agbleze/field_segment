#%%
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

class FieldDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.imgs = sorted(glob(f"{img_dir}/*.tif"))
        self.masks = sorted(glob(f"{mask_dir}/*.tif"))
        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        
        img = get_tiff_img(path=img_path, return_all_bands=False, 
                           bands=("B04", "B03", "B02")
                           )
        mask = torch.tensor(np.array(Image.open(mask_path), dtype=np.int32))
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(masks)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = idx
        area = (boxes[:,3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        img = tv_tensors.Image(img).permute(2,0,1)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", 
                                                   canvas_size=F.get_size(img)
                                                   )
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs)
        
        
        
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer, num_classes
                                                       )
    return model    
        
        




def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)



#%%

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
#dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
img_dir = "/home/lin/codebase/field_segment/data/train_images/images"
mask_dir = "/home/lin/codebase/field_segment/masks"
dataset = FieldDataset(img_dir=img_dir, mask_dir=mask_dir,
                       transforms=get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

# For Training
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)  # Returns losses and detections
print(output)

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)  # Returns predictions
print(predictions[0])

# %%
from engine import train_one_epoch, evaluate
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_classes = 2
dataset = FieldDataset(img_dir=img_dir, mask_dir=mask_dir,
                       transforms=get_transform(train=True)
                       )
dataset_test = FieldDataset(img_dir=img_dir, mask_dir=mask_dir,
                            transforms=get_transform(train=False)
                            )

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-10])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                          shuffle=True,
                                          collate_fn=utils.collate_fn
                                          )


data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 100
model_store_dir = "model_store"
model_name = "fieldmask_net"
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    metric_log, optimizer = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    coco_evaluator = evaluate(model, data_loader_test, device=device, epoch=epoch, 
                              model_store_dir=model_store_dir,
                                model_name=model_name, optimizer=optimizer
                            )

print("That's it!")


#%%

import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# "/home/lin/codebase/field_segment/data/test_images/images/test_0.tif",
test_img_paths = sorted(glob(f"data/test_images/images/*.tif"))
test_img_paths= [test_img_paths[-1]]
for img_path in test_img_paths:
    image = get_tiff_img(path=img_path,
                        return_all_bands=False,
                        bands=("B04", "B03", "B02"))
    eval_transform = get_transform(train=False)
    model_path = "/home/lin/codebase/field_segment/model_store/fieldmask_net_epoch_100.pth"

    model_dict = torch.load(model_path)

    model.load_state_dict(model_dict)

    image = tv_tensors.Image(image).permute(2,0,1)
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]


    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"field: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.5).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))

#%%
import numpy as np
import cv2
import torch
import json

def mask_to_polygon(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Approximate the contour to a polygon and flatten the polygon array
    polygons = [cv2.approxPolyDP(contour, 1, True).flatten().tolist() for contour in contours]
    
    return polygons

# Assume 'masks' is the output from your Mask R-CNN model
#masks = torch.tensor(...)  # Replace with your masks

# Convert each mask to a polygon
annotations = []
for i in range(masks.shape[0]):
    mask = masks[i].cpu().numpy().astype(np.uint8)
    polygons = mask_to_polygon(mask)
    for polygon in polygons:
        annotations.append({'class': i, 'segmentation': polygon})

print(annotations)


#%%
# def predict_segmask(model, image, eval_transform=get_transform(train=False),
#                     device="cuda", pixel_class_proba_threshold=0.5,
#                     visualize=True, mask_transparency=0.3
#                     ):
#     #eval_transform = get_transform(train=False)
#     #model_path = "/home/lin/codebase/field_segment/model_store/fieldmask_net_epoch_100.pth"

#     #model_dict = torch.load(model_path)

#     #model.load_state_dict(model_dict)

#     image = tv_tensors.Image(image).permute(2,0,1)
#     model.eval()
#     with torch.no_grad():
#         if eval_transform:
#             x = eval_transform(image)
#         # convert RGBA -> RGB and move to device
#         x = x[:3, ...].to(device)
#         predictions = model([x, ])
#         pred = predictions[0]

#     #pred_labels = [f"field: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
#     #pred_boxes = pred["boxes"].long()
#     #output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

#     masks = (pred["masks"] > pixel_class_proba_threshold).squeeze(1)
    
#     if visualize:
#         image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
#         image = image[:3, ...]
#         output_image = draw_segmentation_masks(image, masks, alpha=mask_transparency)
#         plt.figure(figsize=(12, 12))
#         plt.imshow(output_image.permute(1, 2, 0))
    
#     return masks    

# #%%  
# import json    
# def create_prediction_annotation(model, test_img_dir,
#                                  img_ext="tif",
#                                 save_as="predicted_segmask.json",
#                                 eval_transform=get_transform(train=False),
#                                 device="cuda", pixel_class_proba_threshold=0.5,
#                                 visualize=True,
#                                 remove_segmnet_less_four_values=True,
#                                 ):
#     test_img_paths = sorted(glob(f"{test_img_dir}/*.{img_ext}"))
    
#     images_annot = []
#     for img_path in test_img_paths:
#         image_name = os.path.basename(img_path)
#         image = get_tiff_img(path=img_path,
#                             return_all_bands=False,
#                             bands=("B04", "B03", "B02")
#                             )
#         #eval_transform = get_transform(train=False)
#         #model_path = "/home/lin/codebase/field_segment/model_store/fieldmask_net_epoch_100.pth"

#         #model_dict = torch.load(model_path)

#         #model.load_state_dict(model_dict)

#         # image = tv_tensors.Image(image).permute(2,0,1)
#         # model.eval()
#         # with torch.no_grad():
#         #     x = eval_transform(image)
#         #     # convert RGBA -> RGB and move to device
#         #     x = x[:3, ...].to(device)
#         #     predictions = model([x, ])
#         #     pred = predictions[0]


#         #image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
#         #image = image[:3, ...]
#         #pred_labels = [f"field: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
#         #pred_boxes = pred["boxes"].long()
#         #output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

#         #masks = (pred["masks"] > threshold).squeeze(1)
#         masks = predict_segmask(model=model, image=image, eval_transform=eval_transform,
#                                 device=device, pixel_class_proba_threshold=pixel_class_proba_threshold,
#                                 visualize=visualize
#                                 )
#         annotations = []
#         for i in range(masks.shape[0]):
#             mask = masks[i].cpu().numpy().astype(np.uint8)
#             polygons = mask_to_polygon(mask)
#             for polygon in polygons:
#                 if remove_segmnet_less_four_values:
#                     if not len(polygon) < 4:
#                         annotations.append({'class': "field", 'segmentation': polygon})
#                 else:
#                     annotations.append({'class': "field", 'segmentation': polygon})

#         print(annotations)
#         img_pred_annt = {"file_name": image_name, "annotations": annotations}
#         images_annot.append(img_pred_annt)

#         # output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")


#         # plt.figure(figsize=(12, 12))
#         # plt.imshow(output_image.permute(1, 2, 0))
#     final_annot = {"images": images_annot}
#     with open(save_as, "w") as file:
#         json.dump(final_annot, file)   
        
#     return final_annot
        
#%%
#test_img_paths = sorted(glob(f"data/test_images/images/*.tif"))

#test_img_dir = "data/test_images/images"
#pred_segmask_annotations = create_prediction_annotation(model=model, test_img_dir=test_img_dir)











#%%

import cv2
import matplotlib.pyplot as plt

# Load the original image
#image = cv2.imread('path_to_your_image.jpg')

# Convert the image from BGR to RGB
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw each polygon on the image
image = get_tiff_img(path=test_img_paths[0],
                        return_all_bands=False,
                        bands=("B04", "B03", "B02"))
for annotation in annotations:
    segmentation = annotation['segmentation']
    # Reshape to (n, 2)
    vertices = np.array(segmentation).reshape(-1, 2)
    # OpenCV requires vertices to be of type int
    vertices = vertices.astype(int)
    # Draw the polygon on the image
    cv2.polylines(image, [vertices], True, (255, 0, 0), 2)

# Display the image
plt.imshow(image)
plt.show()

def create_prediction_annotation(model, test_img_dir,
                                 save_as):
    pass








# %%
import numpy as np
import cv2
import torch
from skimage import measure

def mask_to_polygon(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Approximate the contour to a polygon and flatten the polygon array
    polygons = [cv2.approxPolyDP(contour, 1, True).flatten().tolist() for contour in contours]
    
    return polygons

# Assume 'masks' is the output from your Mask R-CNN model
masks = pred["masks"] # torch.tensor(...)  # Replace with your masks

# Convert each mask to a polygon
annotations = []
for i in range(masks.shape[0]):
    for j in range(masks.shape[1]):
        mask = masks[i, j].cpu().numpy().astype(np.uint8)
        polygons = mask_to_polygon(mask)
        for polygon in polygons:
            annotations.append({'class': j, 'segmentation': polygon})

print(annotations)



#%%
mask = np.array(pred["masks"].to("cpu"))
cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#%%

mask = np.array(pred["masks"].to("cpu"))
# Convert the mask to a polygon
polygon = mask_to_polygon(mask)

print('segmentation:', polygon)




# %%
tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        ...,


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]]], device='cuda:0')