from abc import ABC, abstractmethod
from typing import Self, Optional, Iterable
from functools import partial
from torch import Tensor
from torch import nn
from xpyutils import lazy_property

class BaseDreamModel(ABC, nn.Module):
    
    def __init__(self) -> None:
        
        super().__init__()
        
        self._output = dict()
        self._feature_layers = []
    
    @lazy_property
    @abstractmethod
    def img_height(self) -> int:
        pass
    
    @lazy_property
    @abstractmethod
    def img_width(self) -> int:
        pass
    
    @lazy_property
    @abstractmethod
    def img_channels(self) -> int:
        pass
    
    @lazy_property
    @abstractmethod
    def img_mean(self) -> list[float]:
        pass
    
    @lazy_property
    @abstractmethod
    def img_std(self) -> list[float]:
        pass
    
    @lazy_property
    @abstractmethod
    def feature_layer_names(self) -> list[str]:
        pass
    
    @lazy_property
    def feature_layer_name_to_idx(self) -> dict[str, int]:
        
        return {
            layer_name: idx
            for idx, layer_name in enumerate(self.feature_layer_names)
        }
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    def watch_layers(self, layer_names: Iterable[str]) -> Self:
        
        # convert to a set
        layer_names = set(layer_names)
        
        # take the intersection with the existing layer names
        layer_names = layer_names.intersection(self.feature_layer_names)
        
        for layer_name in layer_names:
            
            # get the layer module
            layer = self.get_feature_layer_by_name(layer_name)
            
            # register a forward hook to record the output of this layer
            layer.register_forward_hook(
                hook=partial(
                    self._record_output, 
                    layer_name=layer_name
                )
            )
        
        return self
    
    def get_output(self, layer_name: str) -> Optional[Tensor]:
        
        return self._output.get(layer_name, None)
        
    def get_feature_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        
        layer_idx = self.feature_layer_name_to_idx.get(layer_name, None)
        
        if layer_idx is None:
            return None
        
        return self._feature_layers[layer_idx]
    
    def _record_output(
            self, 
            module: nn.Module, 
            input: Tensor,
            output: Tensor,
            layer_name: str
        ):
        
        self._output[layer_name] = output
        
        