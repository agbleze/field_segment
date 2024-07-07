
#%%
import os
import torch
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from glob import glob
#from field_segment_utils import get_tiff_img
from PIL import Image
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
import utils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
import rasterio

#%%

from field_segment_utils import visualize_segmask, get_tiff_img, create_instance_segmask

#%%
# def get_tiff_img(path, return_all_bands, bands=("B01", "B03", "B02"),
#                  normalize_bands=True
#                 ):
#     all_band_names = ("B01","B02", "B03","B04","B05", "B06",
#                       "B07","B08","B8A","B09","B11","B12"
#                     )
#     if return_all_bands:
#         band_indexs = [all_band_names.index(band_name) for band_name in all_band_names]
    
#     else:
#         band_indexs = [all_band_names.index(band_name) for band_name in bands]
#     #print(band_indexs)
#     with rasterio.open(path) as src:
#         img_bands = [src.read(band) for band in range(1,13)]
#     dstacked_bands = np.dstack([img_bands[band_index] for band_index in band_indexs])
#     #dstacked_bands = np.dstack([img_bands[3], img_bands[2], img_bands[1]])
#     if normalize_bands:
#         # Normalize bands to 0-255
#         dstacked_bands = ((dstacked_bands - dstacked_bands.min()) / 
#                           (dstacked_bands.max() - dstacked_bands.min()) * 255
#                           ).astype(np.uint8)

#     return dstacked_bands

#%%
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_dir = "/home/lin/codebase/field_segment/data/train_images/images"
mask_dir = "/home/lin/codebase/field_segment/masks_smp"
train_annot_path = "/home/lin/codebase/field_segment/train_annotation.json"



#%%
create_instance_segmask(annotation_path=train_annot_path, img_dir=img_dir, output_dir=mask_dir)


# %%
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
    
#%%
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    #CLASSES = ['field']
    
    def __init__(
            self, 
            images_list, 
            masks_list, 
            #classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.imgs = sorted(images_list) #sorted(glob(f"{images_dir}/*.tif"))
        self.masks = sorted(masks_list) #sorted(glob(f"{masks_dir}/*.tif"))
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        #self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        img_path = self.imgs[i]
        mask_path = self.masks[i]
        new_width = 1216
        new_height = 832
        #print(mask_path)
        image = get_tiff_img(path=img_path, return_all_bands=False, 
                           bands=("B04", "B03", "B02")
                           )
        image = cv2.resize(image, (new_width,new_height), interpolation=cv2.INTER_NEAREST)
        image = image/255.0
        #mask = np.array(Image.open(mask_path), dtype=np.int32) #torch.tensor(np.array(Image.open(mask_path), dtype=np.int32))
        mask = cv2.imread(mask_path)#.astype('float')
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        #mask = cv2.resize(mask, (1216,832))
        #mask = mask/255.0
        # read data
        #image = cv2.imread(self.images_fps[i])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            mask = mask.squeeze().long()
        else:
            image = torch.from_numpy(image).permute(2, 0,1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).squeeze().long()
            
        return image, mask
        
    def __len__(self):
        return len(self.imgs)  
    

#%%  

images_list = sorted(glob(f"{img_dir}/*.tif"))
masks_list = sorted(glob(f"{mask_dir}/*.tif"))

#%%
dataset = Dataset(images_list=images_list, masks_list=masks_list)

image, mask = dataset[4] # get some sample
visualize(image=image,mask=mask.squeeze(),) 

#%%
img_37_path = "/home/lin/codebase/field_segment/masks_smp/mask_train_35.tif"

#%%
mask_raw = cv2.imread(img_37_path).astype('float')   
#cv2.cvtColor(mask, )

#%%

mask.shape

#%%
mask2 = Image.open(img_37_path)
#mask2 = mask2.load()


#%%
img_path = "/home/lin/codebase/field_segment/data/train_images/images/train_35.tif"
img = get_tiff_img(path=img_path, return_all_bands=False, 
                    bands=("B04", "B03", "B02")
                    )

#%%
img
#img.shape

cv2.resize(img, (1216,832)).shape
#img.resize((1216,832))

#%%

np.min(img/255)

img_scaled = img / 255
#%%

Image.fromarray(img)


#%%

#mask/255.0

#%%

np.min(mask/255)


#%%
#%%

#mask = torch.tensor(np.array(Image.open(img_37_path), dtype=np.int32))
# %%
import albumentations as albu

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, border_mode=cv2.BORDER_CONSTANT, value=0),#always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                #albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                #albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(min_height=320, min_width=320, border_mode=cv2.BORDER_CONSTANT, value=0),
        #albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# %%
#### Visualize resulted augmented images and masks

augmented_dataset = Dataset(images_list=images_list, masks_list=masks_list,
                            augmentation=get_training_augmentation(),
                            )

# same image with different random transforms
for i in range(3):
    image, mask = augmented_dataset[1]
    visualize(image=image, mask=mask)
# %%
ENCODER = 'mit_b0' #'se_resnext50_32x4d'
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
# %%
preprocessing_fn.keywords
# %%
train_image_list = images_list[:40]
train_mask_list = masks_list[:40]
valid_image_list = images_list[40:]
valid_mask_list = masks_list[40:]

#%%
train_dataset = Dataset(images_list=sorted(train_image_list), 
                        masks_list=sorted(train_mask_list),
                        augmentation=False,#get_training_augmentation(), 
                        preprocessing=False, #get_preprocessing(preprocessing_fn),
                    )

valid_dataset = Dataset(images_list=sorted(valid_image_list),
                        masks_list=sorted(valid_mask_list),
                        augmentation=False, #get_validation_augmentation(), 
                        preprocessing=False, #get_preprocessing(preprocessing_fn),
                    )

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

#%%
loss = DiceLoss()
metrics = [
    IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

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



# %%
# train model for 40 epochs

max_score = 0

for i in range(0, 40):    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
# %%
