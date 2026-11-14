"""A plain U-Net in PyTorch, written to be read.

Input : [B, 3, H, W]  colour photograph, values 0..1
Output: [B, 1, H, W]  one LOGIT per pixel (any real number, not a probability)

A logit becomes a vessel probability with sigmoid(logit) = 1 / (1 + exp(-logit)).
The sigmoid is NOT inside the model: the loss (BCEWithLogitsLoss) applies it
itself in a numerically safe way.

Shapes for one 512 x 512 crop with base = 32 channels:

    input                      [B,   3, 512, 512]
    enc1  (two 3x3 convs)      [B,  32, 512, 512]  --------- skip 1 ---------+
    pool                       [B,  32, 256, 256]                            |
    enc2                       [B,  64, 256, 256]  ------- skip 2 -------+   |
    pool                       [B,  64, 128, 128]                        |   |
    enc3                       [B, 128, 128, 128]  ----- skip 3 -----+   |   |
    pool                       [B, 128,  64,  64]                    |   |   |
    enc4                       [B, 256,  64,  64]  --- skip 4 ---+   |   |   |
    pool                       [B, 256,  32,  32]                |   |   |   |
    bottleneck                 [B, 512,  32,  32]                |   |   |   |
    upsample + concat skip 4   [B, 768,  64,  64]  <-------------+   |   |   |
    dec4                       [B, 256,  64,  64]                    |   |   |
    upsample + concat skip 3   [B, 384, 128, 128]  <-----------------+   |   |
    dec3                       [B, 128, 128, 128]                        |   |
    upsample + concat skip 2   [B, 192, 256, 256]  <---------------------+   |
    dec2                       [B,  64, 256, 256]                            |
    upsample + concat skip 1   [B,  96, 512, 512]  <-------------------------+
    dec1                       [B,  32, 512, 512]
    head (1x1 conv)            [B,   1, 512, 512]  logits
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """Two times: 3x3 convolution -> batch norm -> ReLU.

    - 3x3 convolution with padding 1: each output pixel is a weighted sum of a
      3x3 neighbourhood. Padding 1 keeps height and width unchanged.
    - Batch norm: rescales each channel to mean 0 / spread 1 over the batch.
      It makes training faster and more stable. (It has its own bias, so the
      convolution does not need one.)
    - ReLU: max(0, x). Without it, many layers would collapse into one linear filter.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net with 4 down steps, a bottleneck and 4 up steps."""

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        # Encoder ("going down"): more channels, smaller picture.
        # Each step sees a wider area of the retina but with less detail.
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        # 2x2 max pooling: keep the strongest value of each 2x2 block -> half the size.
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck: the smallest picture (1/16 size) with the most channels.
        self.bottleneck = DoubleConv(c4, c5)

        # Decoder ("going up"): the input of each block is the upsampled
        # feature map CONCATENATED with the encoder feature map of the same
        # size (the skip connection). That is why the channel counts add up.
        self.dec4 = DoubleConv(c5 + c4, c4)
        self.dec3 = DoubleConv(c4 + c3, c3)
        self.dec2 = DoubleConv(c3 + c2, c2)
        self.dec1 = DoubleConv(c2 + c1, c1)

        # 1x1 convolution: a weighted sum of the 32 channels at each pixel -> 1 logit.
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _up_and_concat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample to exactly the size of the skip tensor (bilinear = smooth
        # blending of neighbours), then stack the channels side by side.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  #              [B,  base, H,    W   ]
        e2 = self.enc2(self.pool(e1))  #  [B, 2base, H/2,  W/2 ]
        e3 = self.enc3(self.pool(e2))  #  [B, 4base, H/4,  W/4 ]
        e4 = self.enc4(self.pool(e3))  #  [B, 8base, H/8,  W/8 ]
        b = self.bottleneck(self.pool(e4))  # [B, 16base, H/16, W/16]

        d4 = self.dec4(self._up_and_concat(b, e4))  #  [B, 8base, H/8, W/8]
        d3 = self.dec3(self._up_and_concat(d4, e3))  # [B, 4base, H/4, W/4]
        d2 = self.dec2(self._up_and_concat(d3, e2))  # [B, 2base, H/2, W/2]
        d1 = self.dec1(self._up_and_concat(d2, e1))  # [B,  base, H,   W  ]
        return self.head(d1)  #                        [B, 1, H, W] logits


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers (weights and biases) in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = UNet(base=32)
    out = net(torch.zeros(1, 3, 512, 512))
    print("input  [1, 3, 512, 512] -> output", list(out.shape))
    print(f"trainable parameters: {count_parameters(net):,}")
