import torch
from torch import nn


class InputAdapter(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=32,
        out_channels=3,
        residual_scale=1.0,
    ):
        super().__init__()
        # print(f"InputAdapter init: in_channels={in_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}, residual_scale={residual_scale}")
        self.residual_scale = residual_scale
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # print(f"InputAdapter forward: image shape: {image.shape}, residual_scale: {self.residual_scale}")
        if image.ndim != 4:
            raise ValueError(f"Expected an NCHW image tensor, got shape {image.shape}")
        return image + self.residual_scale * self.network(image)