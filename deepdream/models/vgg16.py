from torch import Tensor
from torch import nn
from torchvision import models
from typing import Self, Optional, Iterable
from functools import partial
from xpyutils import lazy_property
from .base import BaseDreamModel

class VGG16(BaseDreamModel):
    
    def __init__(self) -> None:
        
        super().__init__()
        
        # load weights
        self._weights = models.VGG16_Weights.DEFAULT

        # load model
        self._model = models.vgg16(weights=self._weights)
    
        # feature layers
        self._feature_layers = self._model.features
        
        # a dictionary that maintains the output tensor from each feature layer
        self._output = dict()
        
    @lazy_property
    def img_height(self) -> int:
        
        return self._weights.transforms().crop_size[0]
    
    @lazy_property
    def img_width(self) -> int:
        
        return self._weights.transforms().crop_size[0]
    
    @lazy_property
    def img_channels(self) -> int:
        
        return 3
    
    @lazy_property
    def img_mean(self) -> list[float]:
        
        return self._weights.transforms().mean
    
    @lazy_property
    def img_std(self) -> list[float]:
        
        return self._weights.transforms().std
        
    @lazy_property
    def feature_layer_names(self) -> list[str]:
        
        return [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'max_pooling1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'max_pooling2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'max_pooling3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'max_pooling4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'max_pooling5',
        ]
        
    def forward(self, x: Tensor) -> Tensor:
        
        out = self._feature_layers(x)
        
        return out

        
    