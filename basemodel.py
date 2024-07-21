
#%%
import torch
import torch.nn as nn
from torchvision.models.resnet import conv1x1 #, conv3x3
#%%
"""
Architecture design  -- option for where to bn before activation or after
option to define activation to use
option to defice indices to use as identity

Stem block
takes 12 channels and output 24 channels
batch norm
activation

-- fradually increase till 48 channels in output

-- gradually decrease to 1 channel
-- add indices channels (NDVI and EVI) 


learner block
-- Learner block should take indices not used in stemblock

task block
"""

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)

class StemModel(nn.Module):
    def __init__(self, in_channels=12, first_out_channel=12, final_out_channels=3):
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=first_out_channel)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(in_channels=12, out_channels=first_out_channel*2)
        self.bn2 = nn.BatchNorm2d(num_features=first_out_channel*2)
        self.conv3 = conv3x3(in_channels=first_out_channel*2, out_channels=first_out_channel*3)
        self.bn3 = nn.BatchNorm2d(num_features=first_out_channel*3)
        self.conv4 = conv3x3(in_channels=first_out_channel*3, out_channels=first_out_channel*4)
        self.bn4 = nn.BatchNorm2d(num_features=first_out_channel*4)
        self.conv5 = nn.Conv2d(in_channels=first_out_channel*4, out_channels=first_out_channel*3)
        self.bn5 = nn.BatchNorm2d(num_features=first_out_channel*3)
        self.conv6 = nn.Conv2d(in_channels=first_out_channel*3, out_channels=first_out_channel*2)
        self.bn6 = nn.BatchNorm2d(num_features=first_out_channel*2)
        self.conv7 = nn.Conv2d(in_channels=first_out_channel*2, out_channels=first_out_channel)
        self.bn7 = nn.BatchNorm2d(num_features=first_out_channel)
        self.conv8 = nn.Conv2d(in_channels=first_out_channel, out_channels=1)
        self.bn8 = nn.BatchNorm2d(num_features=1)
        
    def forward(x, index_1=None, index_2=None):
        # run the convultion and concat the index in the last layer
        


class CombineBlock(nn.Module):
    def __init__(self):
        pass
    
    
# create prestem model and attach to existing model           
 
# create index models
        
##
#
#

class IndexBlock(nn.Module):
    def __init__(self, bands=["B04", "B03", "B02"], identity_index="NDVI"):
        pass
    
    def get_index():
        pass
    
    def calculate_index_map():
        pass
            
    
    def get_projection_link():
        pass
    
    def __len__(self):
        pass
    
    def indexblock_conv():
        pass
    
    
    



