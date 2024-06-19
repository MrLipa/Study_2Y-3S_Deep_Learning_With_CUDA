import math

class LossFunction():
    def __init__(self, numberOfBins = 16):
        '''
        number of bins per channel a and b, best if 256 divided by it is a round number
        '''
        self.numberOfBins = numberOfBins
        self.scaleFactor = int(256/self.numberOfBins)

    def lab2class(self, pixel):
        L, a, b = pixel
        aClass = int(a / self.scaleFactor)
        bClass = int(b / self.scaleFactor)
        return aClass * int(256/self.scaleFactor) + bClass

    def class2ab(self, inClass):
        aClass = int(inClass / int(256/self.scaleFactor))
        a = int(aClass * self.scaleFactor + int(self.scaleFactor/2))
        bClass = inClass - aClass * int(256/self.scaleFactor)
        b = int(bClass * self.scaleFactor + int(self.scaleFactor/2))
        return a, b

    def getWeight(self, targetClass):
        return 1

    def imageEntrophyLoss(self, output, target):
        '''
        output - nn output of shape hw x q
        target - target image
        assumend output of shape (hxw) x q
        length of high*width and depth of number of classes q
        '''
        print(f"output: {output.shape}")
        print(f"target: {target.shape}")
        sum = 0
        for image, groundTrouth in zip(output, target):
            sum += self.oneImageEntrophy(image, groundTrouth)
        return sum / target[0]

    def oneImageEntrophy(self, output, target):
        sum = 0
        for row in range(target.shape[0]):
            for col in range(target.shape[1]):
                pixel = row * target.shape[0] + col
                targetClass = self.lab2class(target[row,col])
                sum += self.getWeight(targetClass) * math.log(output[pixel][targetClass])
        return -sum