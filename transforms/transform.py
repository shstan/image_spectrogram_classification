from torchvision import transforms as T
import torch
import numpy as np
from torchvision.transforms import RandomHorizontalFlip,\
    RandomVerticalFlip, RandomResizedCrop, ColorJitter, RandomApply, Compose

p=0.5

base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class IdentityTransform(object):
    def __call__(self, x):
        return x

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=(0.1, 1.0)):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.normal(mean=self.mean, std=np.random.uniform(*self.std), size=tensor.size()).to(tensor.device)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

augmented_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.RandomHorizontalFlip(p=p),
            T.RandomVerticalFlip(p=p),
            T.RandomApply([T.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0))], p=p),
            T.RandomApply([AddGaussianNoise()], p=p),
    ])

# this augmentation adds some of the random transformations all the time, ensuring that the resulting augmented image is always different in something.
simsiam_representation_transform = T.Compose([
        T.RandomHorizontalFlip(p=p),
        T.RandomVerticalFlip(p=p),
        T.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0)),
        AddGaussianNoise(),
])