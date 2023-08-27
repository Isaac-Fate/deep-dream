from typing import Self
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
from PIL import Image
import numpy as np
import torch
from torch import Tensor

class ImageToTensorTransformer(BaseEstimator, TransformerMixin):
    
    MAX_PIXEL_VALUE = 255
    
    def __init__(
            self, 
            mean: list[float], 
            std: list[float],
            height: int,
            width: int
        ) -> None:
        
        super().__init__()
        
        self.mean = mean
        self.std = std
        self.height = height
        self.width = width
        
    def fit(self, img: Image.Image) -> Self:
        
        self.img_original_width, self.img_original_height = img.size
        
        return self
    
    def transform(self, img: Image.Image) -> Tensor:
        
        # resize image
        desired_shape = (self.height, self.width)
        img = img.resize(desired_shape)
        
        # convert to Numpy array
        img_array = np.array(img)
        
        # scale pixel values to real numbers in between 0 and 1
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32)
            img_array /= self.MAX_PIXEL_VALUE

        # convert mean and std to correct shape
        # so that they are the statistics of color channels
        mean = np.array(self.mean).reshape(1, 1, 3)
        std = np.array(self.std).reshape(1, 1, 3)
        
        # normalize
        img_array = (img_array - mean) / std
        
        # convert to tensor
        img_tensor = torch.tensor(
            img_array, 
            
            # it is necessary to set this type
            dtype=torch.float32,
            
            # requires_grad=True
        ).permute(2, 0, 1).unsqueeze(dim=0)
        
        return img_tensor
    
    def fit_transform(self, img: Image.Image) -> Tensor:
        
        return super().fit_transform(img)
    
    def inverse_transform(self, img_tensor: torch.Tensor) -> Image.Image:
        
        img_array: np.ndarray = img_tensor.squeeze(dim=0) \
            .permute(1, 2, 0) \
            .clone() \
            .detach() \
            .numpy()
        
        # convert mean and std to correct shape
        # so that they are the statistics of color channels
        mean = np.array(self.mean).reshape(1, 1, 3)
        std = np.array(self.std).reshape(1, 1, 3)
        
        # recover
        img_array = img_array * std + mean
        
        # make values in between 0 and 1
        img_array = img_array.clip(0.0, 1.0)
        
        # convert back to pixel values
        img_array *= self.MAX_PIXEL_VALUE
        
        # cast to uint8 data type
        img_array = img_array.astype(np.uint8)
        
        # convert to image
        img = Image.fromarray(img_array)
        
        # resize the image to the orginial scale
        img_original_size = (self.img_original_width, self.img_original_height)
        img = img.resize(img_original_size)
        
        return img
    
    
    