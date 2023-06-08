# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .losses import (L1Loss, MSELoss, PSNRLoss, VGGPerceptualLoss,TextureLossVGG19, KLTStereoSRQA)

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'VGGPerceptualLoss','TextureLossVGG19', 'KLTStereoSRQA'
]