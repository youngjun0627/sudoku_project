import torch
import os
import cv2
from torch.utils.data import Dataset
import csv
import numpy as np
from transform import create_train_transform, create_val_transform

## 이미지당 한개씩이라고 가정
class Custom_Dataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):

        self.transform = transform        
        self.root = root
        self.images = []  
        self.masks = []
        self.masks2 = []
        self.keypoints = []
        self.sudokus = []
        self.solutions = []
        self.mode = mode        
        self.scale =0.45
        with open(os.path.join(root,'{}.csv'.format(mode)), 'r', encoding='utf-8-sig') as f:
            rdr = csv.reader(f)
            for line in rdr:
                image_path = line[0]
                mask_path = line[1]
                mask2_path = line[2]
                sudoku = line[3]
                solution = line[4]
                keypoint = line[5]
                
                self.images.append(image_path)
                self.masks.append(mask_path)
                self.masks2.append(mask2_path)
                self.sudokus.append(sudoku)
                self.solutions.append(solution)
                self.keypoints.append(keypoint)

    def __getitem__(self, index):

        image = self.images[index]        
        imagename = self.images[index]
        image = cv2.imread(image, cv2.IMREAD_COLOR)   
        image = np.array(image)

        mask = self.masks[index]
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask)

        mask2 = self.masks2[index]
        mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)
        mask2 = np.array(mask2)

        keypoint = self.keypoints[index]
        keypoint = np.array(self.list_to_num(keypoint, 'keypoint'))
        if self.transform:        
            image, mask, mask2, keypoint = self.transform(image, mask, mask2, keypoint)
        image = image.float() / 255.
        mask = mask.float() / 255.
        mask2 = mask2.long()
        keypoint = np.array(keypoint)
        keypoint = self.generate_heatmap(image,keypoint)
        keypoint = torch.tensor(keypoint, dtype=torch.float)
        #keypoint = np.array(keypoint).astype('float').reshape(-1)
        return image, mask, mask2, keypoint

    def __len__(self):    
        return len(self.images)    

        return label
    def list_to_num(self, _list, mode):
        _list =  _list.replace('[','')
        _list =  _list.replace(']','')
        _list =  _list.replace(')','')
        _list =  _list.replace('(','')
        if mode=='sudoku' or mode=='solution':
            result =  list(map(int,_list.split(',')))

        elif mode=='keypoint':
            result = []
            _list =  list(map(float, _list.split(',')))
            for i in range(0, len(_list), 2):
                x,y = _list[i], _list[i+1]
                result.append((x,y))
        
        return result

    def generate_heatmap(self, image, keypoint):
        filter_size = 9
        sigma = 0.5
        heatmap = np.zeros((len(keypoint),image.shape[1], image.shape[2]))

        #keypoint = keypoint.astype('int')

        #kernel = cv2.getGaussianKernel(filter_size, sigma)
        for i,key in enumerate(keypoint):
            
            if i==0:
                heatmap[i,max(int(key[1])-1,0),max(int(key[0])-1,0)] = 1
            if i==1:
                heatmap[i,max(int(key[1])-1,i),min(int(key[0])+1,image.shape[2]-1)] = 1
            if i==2:
                heatmap[i,min(int(key[1])+1,image.shape[1]-1),max(int(key[0])-1, 0)] = 1
            if i==3:
                heatmap[i,min(int(key[1])+1, image.shape[1]-1),min(int(key[0])+1, image.shape[2]-1)] = 1

            
            #heatmap[i,:,:] = cv2.GaussianBlur(heatmap[i,:,:], (filter_size, filter_size), sigma)
            #heatmap[i,:,:] /= heatmap[i,:,:].max()
        return heatmap
if __name__=='__main__':
    dataset = Custom_Dataset('.', mode ='train', transform = create_val_transform(600,0.5))
    #print(np.max(Custom_Dataset('../')[0][1]))
    for i in range(len(dataset)):
        img = dataset[i][0]
        img = (img.squeeze(0).permute(1,2,0).cpu().detach().numpy() * 255.).astype('uint8')
        keypoint = dataset[i][3]
        print(keypoint.shape)
        #print(keypoint[np.argmax(keypoint)])
        vertex = []
        for idx,key in enumerate(keypoint):
            key = key.numpy()
            cv2.normalize(key, key, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}.jpg'.format(idx), key.astype('float'))
            index = np.argmax(key).tolist()
            vertex.append((index//300, index%300))
            print(key[index//300, index%300])
            img = cv2.circle(img, ((index%300), (index//300)), 10, (255,0,0), 3)
        cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}.jpg'.format('result'), img)
        print(vertex)
        #print(dataset[i][3].shape)
        if i==5:
            break
        '''
        print(dataset[i][0].shape)
        
        print(dataset[i][1].shape)
        print(dataset[i][2].shape)
        print(dataset[i][3].shape)
        '''
        '''
        for a in dataset[i][3]:
            for i in range(270):
                if a[i]!=0:
                    print(a[i])    
        cv2.imwrite('aa.png', (dataset[i][3] * 255.).astype('uint8'))
        break
        '''
        #print(dataset[i][1].max())
        #print(dataset[i][1].min())

