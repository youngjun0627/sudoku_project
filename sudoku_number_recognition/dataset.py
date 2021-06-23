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
                sudoku = line[3]
                solution = line[4]
                keypoint = line[5]
                
                self.images.append(image_path)
                self.masks.append(mask_path)
                self.sudokus.append(sudoku)
                self.solutions.append(solution)
                self.keypoints.append(keypoint)

    def __getitem__(self, index):

        image = self.images[index]        
        imagename = self.images[index]
        image = cv2.imread(image, cv2.IMREAD_COLOR)   

        keypoint = self.keypoints[index]
        keypoint = np.array(self.list_to_num(keypoint, 'keypoint'))
        '''
        for key in keypoint:
            image = cv2.circle(image, (int(key[0]), int(key[1])), 10,(255,0,0))
        cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}1.jpg'.format(i), image)
        '''
        if self.transform:        
            image = self.transform(image, keypoint)
        image = image.float() / 255.

        sudoku = self.sudokus[index]
        sudoku = np.array(self.list_to_num(sudoku, 'sudoku'), dtype=np.long).reshape(9,9)
        return image, sudoku

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
    dataset = Custom_Dataset('.', mode ='train', transform = create_val_transform(600,0.45))
    #print(np.max(Custom_Dataset('../')[0][1]))
    for i in range(len(dataset)):
        img = dataset[i][0]
        img = (img.permute(1,2,0).cpu().detach().numpy() * 255.).astype('uint8')

        cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}.jpg'.format(i), img)
        if i==1:
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

