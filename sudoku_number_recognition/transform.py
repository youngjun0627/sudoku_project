import random
import cv2
import numpy as np
import albumentations
from albumentations.pytorch import ToTensorV2

class create_train_transform(object):

    def __init__(self, original_size, scale):
        super(create_train_transform, self).__init__()
        self.size = int(original_size*scale)
        self.transforms = []
        self.transforms += [albumentations.Resize(self.size, self.size, interpolation=2)]
        self.transforms = albumentations.Compose(self.transforms,  keypoint_params=albumentations.KeypointParams(format='xy'))
        self.totensor = albumentations.Compose([ToTensorV2()])
    def __call__(self, image, keypoint):
        tr = self.transforms(image = image, keypoints = keypoint)
        image = tr['image']
        srcpoint = tr['keypoints']
        #srcpoint = keypoint.tolist()
        left_top = srcpoint[0]
        right_top = srcpoint[1]
        right_down = srcpoint[3]
        left_down = srcpoint[2]
        '''
        srcpoint = np.array([
                    left_top,
                    right_top,
                    right_down,
                    left_down
                ], dtype=np.float32)
        '''
        srcpoint = np.array([
                [max(left_top[0]-random.randint(0,10),0), max(left_top[1]-random.randint(0,10),0)],
                [min(right_top[0]+random.randint(0,10), self.size), max(right_top[1]-random.randint(0,10),0)],
                [min(right_down[0]+random.randint(0,10),self.size), min(right_down[1]+random.randint(0,10),self.size)],
                [max(left_down[0]-random.randint(0,10),0), min(left_down[1]+random.randint(0,10),self.size)]
                ], dtype=np.float32)
        
        dstpoint = np.array([[0,0], [self.size,0], [self.size, self.size], [0, self.size]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(srcpoint, dstpoint)
        image = cv2.warpPerspective(image, matrix, (self.size, self.size))
        image = self.totensor(image=image)['image']
        return image

class create_val_transform(object):

    def __init__(self, original_size, scale):
        super(create_val_transform, self).__init__()
        self.size = int(original_size*scale)
        self.transforms = []
        self.transforms += [albumentations.Resize(self.size, self.size, interpolation=2)]
        self.transforms = albumentations.Compose(self.transforms,  keypoint_params=albumentations.KeypointParams(format='xy'))
        self.totensor = albumentations.Compose([ToTensorV2()])
    def __call__(self, image, keypoint):
        tr = self.transforms(image = image, keypoints = keypoint)
        image = tr['image']
        srcpoint = tr['keypoints']
        #srcpoint = keypoint.tolist()
        left_top = srcpoint[0]
        right_top = srcpoint[1]
        right_down = srcpoint[3]
        left_down = srcpoint[2]
        srcpoint = np.array([
                    left_top,
                    right_top,
                    right_down,
                    left_down
                ], dtype=np.float32)
        dstpoint = np.array([[0,0], [self.size,0], [self.size, self.size], [0, self.size]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(srcpoint, dstpoint)
        image = cv2.warpPerspective(image, matrix, (self.size, self.size))
        image = self.totensor(image=image)['image']
        return image
