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
mask_dir = "/home/lin/codebase/field_segment/masks_smp"
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

#%% For inference
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

#%% get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
#device="cpu"
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params,
    lr=0.005,
    #momentum=0.9,
    weight_decay=0.0005) 


# and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=3,
#     gamma=0.1
# )

#%% let's train it just for 2 epochs
num_epochs = 100
model_store_dir = "model_store_binary"
model_name = "fieldmask_net_binary"
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    metric_log, optimizer = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    #lr_scheduler.step()
    # evaluate on the test dataset
    coco_evaluator = evaluate(model, data_loader_test, device=device, epoch=epoch, 
                              model_store_dir=model_store_dir,
                                model_name=model_name, optimizer=optimizer
                            )

print("That's it!")



#%% train with segmentation models pytorch
import torch
import numpy as np
import segmentation_models_pytorch as smp

ENCODER = 'resnet34' #'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['field']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, #in_channels=12, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

#%%
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = DiceLoss()
metrics = [
    IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

#%%

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

#%%
# train model for 40 epochs

max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(data_loader)
    valid_logs = valid_epoch.run(data_loader_test)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


#%%

# Take epoch 1 to be best and test on images

#%%

import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# "/home/lin/codebase/field_segment/data/test_images/images/test_0.tif",
test_img_paths = sorted(glob(f"data/test_images/images/*.tif"))






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