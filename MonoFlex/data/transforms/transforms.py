import torch
from torchvision.transforms import functional as F


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, prev_image, target):
        for t in self.transforms:
            image, prev_image, target = t(image, prev_image, target)
        return image, prev_image, target

class ToTensor():
    def __call__(self, image, prev_image, target):
        return F.to_tensor(image), F.to_tensor(prev_image), target

class Normalize():
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image, prev_image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        prev_image = F.normalize(prev_image, mean=self.mean, std=self.std)


        if self.to_bgr:
            image = image[[2, 1, 0]]
            prev_image = prev_image[[2,1,0]]
        
        return image, prev_image, target
