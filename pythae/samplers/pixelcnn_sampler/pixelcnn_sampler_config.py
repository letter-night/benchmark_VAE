from pydantic.dataclasses import dataclass

from ..base import BaseSamplerConfig


@dataclass
class PixelCNNSamplerConfig(BaseSamplerConfig):
    """This is the PixelCNN sampler configuration instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
        n_layers (int): The number of convolutional layers in the model. Default: 10.
        kernel_size (int): The kernel size in the convolutional layers. It must be odd. Default: 5
    """

    n_layers: int = 10
    kernel_size: int = 5

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.kernel_size % 2 == 1
        ), f"Wrong kernel size provided. The kernel size must be odd. Got {self.kernel_size}."