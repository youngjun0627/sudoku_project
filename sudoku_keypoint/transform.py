import albumentations
from albumentations.pytorch import ToTensorV2

class create_train_transform(object):

    def __init__(self, original_size, scale):
        super(create_train_transform, self).__init__()
        self.size = int(original_size*scale)
        self.img_translist = []
        self.img_translist += [albumentations.SafeRotate(limit=30, interpolation=2, border_mode=3)]
        self.img_translist += [albumentations.Blur(blur_limit=5)]
        self.img_translist += [albumentations.MotionBlur(blur_limit = 5)]
        self.img_translist += [albumentations.ColorJitter()]
        self.img_translist += [albumentations.MedianBlur(blur_limit=5)]
        self.img_translist += [albumentations.Resize(self.size, self.size, interpolation=2)]        
        self.img_translist += [ToTensorV2()]        
        self.img_translist = albumentations.Compose(self.img_translist, 
                                        additional_targets = {'mask2': 'mask'},
                                        keypoint_params=albumentations.KeypointParams(format='xy'))
        
    def __call__(self, image, mask, mask2, keypoint):
        tr = self.img_translist(image = image, mask=mask, mask2=mask2, keypoints = keypoint)
        image = tr['image']
        mask = tr['mask']
        mask2 = tr['mask2']
        image = tr['image']        
        keypoint = tr['keypoints']
        return image, mask, mask2, keypoint


class create_val_transform(object):

    def __init__(self, original_size, scale):
        super(create_val_transform, self).__init__()
        self.size = int(original_size * scale)
        self.img_translist = [albumentations.Resize(self.size, self.size, interpolation=2)]        
        #self.img_translist += [albumentations.Normalize()]        
        self.img_translist += [ToTensorV2()]        
        self.img_translist = albumentations.Compose(self.img_translist,
                 additional_targets = {'mask2': 'mask'},
                 keypoint_params=albumentations.KeypointParams(format='xy'))
    def __call__(self, image, mask, mask2, keypoint):
        tr = self.img_translist(image = image, mask=mask, mask2=mask2, keypoints = keypoint)
        image = tr['image']
        mask = tr['mask']
        mask2 = tr['mask2']
        keypoint = tr['keypoints']
        return image, mask, mask2, keypoint
