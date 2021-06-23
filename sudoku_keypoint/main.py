import torch
from model import Keypoint_Net
from loss import Loss, Custom_CrossEntropyLoss
from dataset import Custom_Dataset
import cv2
from transform import create_train_transform, create_val_transform
from torch.utils.data import DataLoader
from activate import train, valid, test
from adamp import AdamP
import argparse
import torch.optim as optim
import numpy as np
import os
from collate_function import CF
import torch.nn as nn

parser = argparse.ArgumentParser(description='sudoku')
parser.add_argument('--mode', default = 'train', type=str, help='insert mode')
parser.add_argument('--imagepath', default = '..', type=str)
args = parser.parse_args()


def main_train(root):
    dataset_root = root

    ### hyper parameter ###
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 300
    BATCHSIZE = 16
    ### custom ###
    save_path = '../key.pth'
    device = torch.device('cuda:0')

    original_size = 600
    scale = 0.45
    train_transform = create_train_transform(original_size, scale)
    val_transform = create_val_transform(original_size, scale)

    trainset = Custom_Dataset(root, transform = train_transform, mode = 'train')
    validset = Custom_Dataset(root, transform = val_transform, mode = 'validation')
    #testset = Custom_Dataset(root, transform = val_transform, mode = 'test')

    train_dataloader = DataLoader(trainset, batch_size = BATCHSIZE, shuffle=True, num_workers=2, collate_fn = CF)
    valid_dataloader = DataLoader(validset, batch_size = BATCHSIZE, shuffle=False, collate_fn = CF)
    mask_weight = [4.62525631, 4.63220993, 4.63623119, 4.63813273, 4.63734552, 4.63742598,
         4.63383315, 4.62894428, 4.62213573, 4.63426785, 4.63924746, 4.64331922,
          4.64468686, 4.64476396, 4.64330268, 4.64006579, 4.63477166, 4.62879268,
           4.63725578, 4.64267293, 4.64451923, 4.6478214 , 4.64677867, 4.64555576,
            4.64252766, 4.63867713, 4.63163978, 4.63890421, 4.64392846, 4.64701173,
             4.64815664, 4.64913104, 4.64764525, 4.64488142, 4.64066539, 4.63524364,
              4.63905957, 4.64444317, 4.64743906, 4.64927723, 4.6475292 , 4.64775819,
               4.64493058, 4.63995064, 4.6342117 , 4.63864049, 4.64234519, 4.64614746,
                4.6465736 , 4.64863083, 4.64625205, 4.64328147, 4.6386988 , 4.63193255,
                 4.63628223, 4.63970538, 4.64245891, 4.64364554, 4.64592485, 4.64320856,
                  4.6419927 , 4.63618325, 4.63108777 ,4.63331975, 4.63681897, 4.63899557,
                   4.64048255, 4.63987318, 4.63898215 ,4.63620129, 4.63207252, 4.62791412,
                    4.63220787, 4.63197887, 4.63307571 ,4.63381924, 4.63471654, 4.63420346,
                     4.63341604, 4.6308558,  4.63034975 ,0.01549217]
    seg_criterion1 = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(2.).to(device))
    seg_criterion2 = nn.CrossEntropyLoss(torch.tensor(mask_weight, dtype=torch.float).to(device))
    key_criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(100.).to(device))
    #key_criterion = Loss()
    model = Keypoint_Net().to(device)
    #model = UNetWithResnet50Encoder().to(device)
    #model = Net().to(device)
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    

    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay = weight_decay, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones = [100,140], gamma=0.5)INUscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, min_lr = 0.000001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, min_lr = 0.000001)
    pre_score = 0
    for epoch in range(epochs):
        train_loss, seg1, seg2, score = train(model, seg_criterion1, seg_criterion2, key_criterion, optimizer, train_dataloader, device)
        print("Epoch : %d Loss: %.4f ..  seg1: %.4f .. seg2: %.4f .. key: %.5f.. LR : %.7f" % (epoch, train_loss, seg1,seg2, score, optimizer.param_groups[0]['lr']))
        if (epoch+1)%5==0:
            valid_loss, seg1, seg2, score = valid(model, seg_criterion1, seg_criterion2, key_criterion, valid_dataloader, device)
            print("Valid Loss: %.4f .. seg1: %.4f ..  seg2: %.4f .. key: %.5f" % (valid_loss, seg1,seg2,score))
            if pre_score<score:
                pre_score=score
                model = model.cpu()
                torch.save(model.state_dict(), save_path)
                model = model.to(device)
            scheduler.step(score)

def main_test(root):

    dataset_root = root

    BATCHSIZE = 1
    ### custom ###
    save_path = '../key.pth'
    device = torch.device('cuda:1')

    assert os.path.exists(save_path)

    original_size = 600
    scale = 0.45

    test_transform = create_val_transform(original_size, scale)

    testset = Custom_Dataset(root, transform = test_transform, mode = 'test')

    
    test_dataloader = DataLoader(testset, batch_size = BATCHSIZE, shuffle=False)

    model = Keypoint_Net().to(device)
    model.load_state_dict(torch.load(save_path))

    test(model, test_dataloader, device)

def predict(img_path):

    BATCHSIZE = 1
    ### custom ###
    save_path = '../key.pth'
    device = torch.device('cuda:0')

    assert os.path.exists(save_path)


    #model = UNetWithResnet50Encoder().to(device)
    model = Keypoint_Net().to(device)
    model.load_state_dict(torch.load(save_path))

    model = model.to(device)
    model.eval()
    original_size = 600
    scale = 0.45
    '''
    root = '.'
    val_transform = create_val_transform(original_size, scale)
    validset = Custom_Dataset(root, transform = val_transform, mode = 'validation')
    valid_dataloader = DataLoader(validset, batch_size = BATCHSIZE, shuffle=False, collate_fn = CF)
    for img, mask, mask2, key in valid_dataloader:
        img = img.to(device)
        for k in key[0]:
            print('a',k.max())
        _,_,output = model(img)
        #print(output)
        output = (output.squeeze(0).cpu().detach().numpy())
        #print(output)
        #cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}.jpg'.format('result'), _img)
        vertex = []
        for key in output:
            #key = key.reshape(-1)
            index = np.argmax(key)
            print(key[(index%270, index//270)])
            vertex.append((index%270, index//270))
        print(vertex)
   
        _img = (img.squeeze(0).permute(1,2,0).cpu().detach().numpy() * 255.).astype('uint8') 
        for node in vertex:
            x, y = node[0], node[1]
            print(x,y)
            _img = cv2.circle(_img, (int(x), int(y)), 10, (255,0,0), 3)
        #print(_img)
        cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}1.jpg'.format('result'), _img)
        break
    '''
    test_transform = create_val_transform(original_size, scale)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(int(original_size*scale), int(original_size*scale)), cv2.INTER_CUBIC)
    img = torch.tensor(img, dtype=torch.float).permute(2,0,1).unsqueeze(0).to(device)
    img = img / 255.

    _,_,output = model(img)
    #print(output)
    output = (output.squeeze(0).cpu().detach())
    #cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}.jpg'.format('result'), _img)
    vertex = []
    for key in output:
        key = key.reshape(-1)
        index = torch.argmax(key)
        vertex.append((index%270, index//270))
   
    _img = (img.squeeze(0).permute(1,2,0).cpu().detach().numpy() * 255.).astype('uint8') 
    for node in vertex:
        x, y = node[0], node[1]
        _img = cv2.circle(_img, (int(x), int(y)), 10, (255,0,0), 3)
    #print(_img)
    cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/result.jpg', _img)
        
if __name__=='__main__':
    imgpath = args.imagepath

    if args.mode =='train':
        main_train(imgpath)
    elif args.mode =='test':
        main_test(imgpath)
    elif args.mode == 'predict':
        predict(imgpath)
