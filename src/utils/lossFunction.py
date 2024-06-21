import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import interpolate

class LossFunction(_Loss):
    def __init__(self, numberOfBins = 16):
        '''
        number of bins per channel a and b, best if 256 divided by it is a round number
        '''
        self.numberOfBins = numberOfBins
        self.scaleFactor = int(256/self.numberOfBins)
        super().__init__()

    def lab2class(self, pixel):
        L, a, b = pixel
        aClass = int(a / self.scaleFactor)
        bClass = int(b / self.scaleFactor)
        return aClass * int(256/self.scaleFactor) + bClass

    def getWeight(self, targetClass):
        return 1

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        output - nn output of shape hw x q
        target - target image
        assumend output of shape (hxw) x q
        length of high*width and depth of number of classes q
        '''
        target = torch.permute(target, (0,3,2,1)) # permute target image to (batchSize, channels, H, W)
        target = interpolate(target, scale_factor=0.25, mode='bilinear') # downsample image to net output size
        target = torch.permute(target, (0,2,3,1)) # permute image to (batchSize, H, W, classes) to match target
        target = torch.flatten(target, start_dim=1, end_dim=2) #flatten for speed

        output = torch.permute(output, (0,2,3,1)) # permute net output to (batchSize, H, W, classes) to match target
        output = torch.flatten(output, start_dim=1, end_dim=2) #flatteb for speed
        sum = 0
        for image, groundTrouth in zip(output, target):
            sum += self.oneImageEntrophy(image, groundTrouth)
        sum = sum / target.size()[0]
        return torch.Tensor(sum)

    def oneImageEntrophy(self, image, groundTrouth):
        sum = 0
        for classes, pixel in zip(image, groundTrouth):
            targetClass = self.lab2class(pixel)
            sum += self.getWeight(targetClass) * classes[targetClass].log()
        return -sum