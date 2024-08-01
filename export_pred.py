
#%%
from prediction_utils import create_prediction_annotation, mask_to_polygon, predict_segmask
from glob import glob
from model_obj import get_transform, get_model_instance_segmentation
import torch
from field_segment_utils import visualize_segmask

#%%

test_img_paths = sorted(glob(f"data/test_images/images/*.tif"))
eval_transform = get_transform(train=False)
model_path = "model_store/fieldmask_net_epoch_100.pth"
model = get_model_instance_segmentation(num_classes=2)

# move model to the right device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model_dict = torch.load(model_path)

model.load_state_dict(model_dict)

test_img_dir = "data/test_images/images"
pred_segmask_annotations = create_prediction_annotation(model=model, 
                                                        test_img_dir=test_img_dir,
                                                        pixel_class_proba_threshold=0.7,
                                                        save_as="predicted_segmask_07_proba.json",
                                                        )

#%%

visualize_segmask(annotation_path="predicted_segmask_07_proba.json",
                  img_dir=test_img_dir
                  )

