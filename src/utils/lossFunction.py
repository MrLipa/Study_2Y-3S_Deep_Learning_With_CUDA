# src/utils/lossFunction.py

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import interpolate


class LossFunction(_Loss):
    def __init__(self, numberOfBins=16):
        self.numberOfBins = numberOfBins
        self.scaleFactor = int(256 / self.numberOfBins)
        super().__init__()

    def lab2class(self, pixel):
        L, a, b = pixel
        aClass = int(a / self.scaleFactor)
        bClass = int(b / self.scaleFactor)
        return aClass * int(256 / self.scaleFactor) + bClass

    def get_weight(self, targetClass):
        return 1

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        output - nn output of shape hw x q
        target - target image
        assumend output of shape (hxw) x q
        length of high*width and depth of number of classes q
        '''
        target = target.float()
        target = interpolate(target.permute(0, 3, 2, 1), scale_factor=0.25, mode='bilinear')
        target = target.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        output = output.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        total_loss = 0
        for image, groundTruth in zip(output, target):
            total_loss += self.one_image_entropy(image, groundTruth)
        mean_loss = total_loss / target.size()[0]
        return torch.Tensor(mean_loss)

    def one_image_entropy(self, image, groundTruth):
        entropy_sum = 0
        for classes, pixel in zip(image, groundTruth):
            targetClass = self.lab2class(pixel)
            entropy_sum += self.get_weight(targetClass) * torch.log(classes[targetClass])
        return -entropy_sum
