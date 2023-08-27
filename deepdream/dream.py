from typing import Optional
from PIL import Image
import logging
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import torch
from torch import Tensor
from torch import nn
from .models.base import BaseDreamModel
from .img2tensor import ImageToTensorTransformer

DREAM_INDEX = 0
TARGET_INDEX = 1
STYLE_INDEX = 2

class DreamMaker:
    
    def __init__(self, model: BaseDreamModel) -> None:
        
        # the underlying computer vission model
        self._model = model
        
        # a transformer that convert image to tensor
        # and it can also recover image from the given tensor
        self._img_to_tensor_transformer = ImageToTensorTransformer(
            mean=self._model.img_mean,
            std=self._model.img_std,
            height=self._model.img_height,
            width=self._model.img_width
        )
        
        self._with_style = False
        
        self._target_and_style_tensor: Optional[Tensor] = None
        
        self._dream_array: Optional[np.ndarray] = None
        
        self._dream_grad: Optional[np.ndarray] = None
        
    @property
    def with_style(self) -> bool:
        
        return self._with_style
    
    def find_content_loss(
            self,
            dream_feature_maps: Tensor,
            target_feature_maps: Tensor
        ) -> Tensor:
        
        # MSE loss function
        loss_fn = nn.MSELoss(reduction="mean")
        
        # find the MSE loss of the feature maps between
        # dream and target images
        loss = loss_fn(dream_feature_maps, target_feature_maps)
        
        return loss
    
    @staticmethod
    def build_gram_matrix(feature_maps: torch.Tensor): 
    
        feature_matrix = feature_maps.flatten(start_dim=1)
        
        gram_matrix = feature_matrix @ feature_matrix.T
        
        return gram_matrix
    
    def find_style_loss(
            self,
            dream_feature_maps,
            style_feature_maps,
        ):
    
        dream_gram_matrix = self.build_gram_matrix(dream_feature_maps)
        style_gram_matrix = self.build_gram_matrix(style_feature_maps)
        img_area = self._model.img_height * self._model.img_width
        
        return torch.sum(torch.square(dream_gram_matrix - style_gram_matrix)) \
            / (4 * self._model.img_channels**2 + img_area**2)
            
    def find_total_variation_loss(
            self,
            dream_tensor: torch.Tensor,
        ):
        
        a = torch.square(
            dream_tensor[:, :, :self._model.img_height - 1, :self._model.img_width - 1]
            - dream_tensor[:, :, 1:, :self._model.img_width - 1]
        )
        
        b = torch.square(
            dream_tensor[:, :, :self._model.img_height - 1, :self._model.img_width - 1]
            - dream_tensor[:, :, :self._model.img_height - 1, 1:]
        )
        
        return torch.sum(torch.pow(a + b, 1.25))
    
    def find_total_loss(
            self,
            dream_tensor: torch.Tensor, 
        ) -> torch.Tensor:
        
        # content loss
        
        content_layer_name = 'relu4_2'
        content_loss_weight = 0.05
        dream_feature_maps: torch.Tensor = self._model.get_output(content_layer_name)[DREAM_INDEX, :, :, :]
        target_feature_maps: torch.Tensor = self._model.get_output(content_layer_name)[TARGET_INDEX, :, :, :]
        
        content_loss = content_loss_weight * self.find_content_loss(
            dream_feature_maps,
            target_feature_maps
        )
        
        # style loss
        
        style_layer_name_to_weight = {
            'relu1_2': 0.1,
            'relu2_2': 0.15,
            'relu3_3': 0.2,
            'relu4_3': 0.25,
            'relu5_3': 0.3
        }
        
        total_style_loss = torch.zeros(size=(1,))
        
        for layer_name, weight in style_layer_name_to_weight.items():
            
            dream_feature_maps: torch.Tensor = self._model.get_output(layer_name)[DREAM_INDEX, :, :, :]
            style_feature_maps: torch.Tensor = self._model.get_output(layer_name)[STYLE_INDEX, :, :, :]
            
            style_loss = self.find_style_loss(
                dream_feature_maps,
                style_feature_maps,
            )
            
            total_style_loss += weight * style_loss
        
        # total variation loss
        
        total_variation_weight = 1e-4
        
        total_variation_loss = total_variation_weight * self.find_total_variation_loss(
            dream_tensor=dream_tensor
        )
        
        # total loss
        total_loss = content_loss + total_style_loss + total_variation_loss
        
        return total_loss
    
    def make_dream(
            self, 
            target_img: Image.Image, 
            style_img: Optional[Image.Image] = None,
            return_loss: bool = False
        ):
        
        # convert target image to tensor using the `fit_transform` method
        target_tensor = self._img_to_tensor_transformer.fit_transform(target_img)
        
        if style_img is None:
            
            self._with_style = False
            
            self._target_and_style_tensor = target_tensor
            
        else:
            
            self._with_style = True
            
            style_tensor = self._img_to_tensor_transformer.transform(style_img)
            
            self._target_and_style_tensor = torch.concat((
                target_tensor,
                style_tensor
            ))
            
        if self._dream_array is None:
            
            # create the initial dream tensor
            dream_tensor = target_tensor.clone()
            
            # convert the dream tensor to a flattened 1D vector
            x0 = dream_tensor.flatten().numpy()
        
        else:
            
            x0 = self._dream_array
            
            
        # make the dream by minimizing the loss function
        # using the L-BFGS-B algorithm
        x: np.ndarray
        x, loss, _ = fmin_l_bfgs_b(
            func=self._find_loss,
            x0=x0,
            fprime=self._find_grad,
            maxfun=20
        )
        
        # record the latest updated dream array x
        self._dream_array = x
        
        # recover the tensor from the 1D vector x
        dream_tensor = torch.from_numpy(x.astype(np.float32)).view(
            1,
            self._model.img_channels,
            self._model.img_height,
            self._model.img_width
        )
        
        # recover the image from the tensor
        dream_img = self._img_to_tensor_transformer.inverse_transform(dream_tensor)
        
        if return_loss:
            return dream_img, loss
        
        return dream_img
    
    def _find_loss(self, x: np.ndarray) -> float:
        
        # convert to dream tensor
        dream_tensor = torch.from_numpy(x.astype(np.float32)).view(
            1,
            self._model.img_channels,
            self._model.img_height,
            self._model.img_width
        ).requires_grad_(True)
        
        # create the input tensor
        assert self._target_and_style_tensor is not None
        input_tensor = torch.concat((
            dream_tensor,
            self._target_and_style_tensor
        ))
        
        # feed forward through the CV model
        self._model(input_tensor)
        
        # compute the total loss
        loss = self.find_total_loss(dream_tensor)
        
        # update the gradients
        loss.backward()
        
        # get the gradient of the dream tensor
        dream_grad = dream_tensor.grad
        
        # convert the gradient to a 1D vector
        # * It is critical to cast to the type `np.float64`!!!
        dream_grad = dream_grad.flatten().numpy().astype(np.float64)
        
        # store the value of the gradient
        self._dream_grad = dream_grad
        
        # convert to a single float number
        loss = loss.detach().numpy().astype(np.float64).item()
        
        return loss
    
    def _find_grad(self, x: np.ndarray) -> np.ndarray:
        
        return self._dream_grad

def make_dream(
        target_img: Image.Image,
        style_img: Image.Image,
        dream_maker: DreamMaker,
        n_epochs: int = 5
    ) -> Image.Image:
    
    for epoch in range(n_epochs):
    
        dream_img, loss = dream_maker.make_dream(
            target_img=target_img,
            style_img=style_img,
            return_loss=True
        )
        
        logging.info(f"Epoch: {epoch + 1} Loss: {loss}")
        
    return dream_img
