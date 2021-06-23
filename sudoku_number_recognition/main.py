import torch
from model import Number_Net
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
from config import params

parser = argparse.ArgumentParser(description='sudoku')
parser.add_argument('--mode', default = 'train', type=str, help='insert mode')
parser.add_argument('--imagepath', default = '..', type=str)
args = parser.parse_args()


def main_train(root):
    dataset_root = root

    ### hyper parameter ###
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    epochs = params['epochs']
    BATCHSIZE = params['batch_size']
    ### custom ###
    save_path = params['num_model']
    device = torch.device('cuda:{}'.format(params['gpu']))

    original_size = 600
    scale = 0.45
    train_transform = create_train_transform(original_size, scale)
    val_transform = create_val_transform(original_size, scale)

    trainset = Custom_Dataset(root, transform = train_transform, mode = 'train')
    validset = Custom_Dataset(root, transform = val_transform, mode = 'validation')
    #testset = Custom_Dataset(root, transform = val_transform, mode = 'test')

    train_dataloader = DataLoader(trainset, batch_size = BATCHSIZE, shuffle=True, num_workers=params['num_workers'], collate_fn = CF)
    valid_dataloader = DataLoader(validset, batch_size = BATCHSIZE, shuffle=False, collate_fn = CF)
    weights = [4. for _ in range(10)]
    weights[0] = 0.3
    criterion = Custom_CrossEntropyLoss(weight = torch.tensor(weights).to(device))
    model = Number_Net().to(device)
    
    '''
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    '''

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, min_lr = 0.000001)
    pre_score = 0
    for epoch in range(epochs):
        train_loss, score = train(model, criterion, optimizer, train_dataloader, device)
        print("Epoch : %d Loss: %.4f ..  score: %.5f.. LR : %.7f" % (epoch, train_loss, score, optimizer.param_groups[0]['lr']))
        if (epoch+1)%5==0:
            valid_loss, score = valid(model, criterion, valid_dataloader, device)
            print("Valid Loss: %.4f .. score: %.5f" % (valid_loss, score))
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
    save_path = params['num_model']
    device = torch.device('cuda:{}'.format(params['gpu']))

    assert os.path.exists(save_path)

    original_size = 600
    scale = 0.45

    test_transform = create_val_transform(original_size, scale)

    testset = Custom_Dataset(root, transform = test_transform, mode = 'test')

    
    test_dataloader = DataLoader(testset, batch_size = BATCHSIZE, shuffle=False)

    model = Number_Net().to(device)
    model.load_state_dict(torch.load(save_path))

    score = test(model, test_dataloader, device)
    print('score : {}'.format(score))

def predict(img_path):

    BATCHSIZE = 1
    ### custom ###
    save_path = params['num_model']
    device = torch.device('cuda:{}'.format(params['gpu']))

    assert os.path.exists(save_path)


    model = Number_Net().to(device)
    model.load_state_dict(torch.load(save_path))

    model = model.to(device)
    model.eval()
    original_size = 600
    scale = 0.45
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
        index = torch.argmin(key)
        print(key[index])
        index = torch.argmax(key)
        print(key[index])
        vertex.append((index%270, index//270))
   
    _img = (img.squeeze(0).permute(1,2,0).cpu().detach().numpy() * 255.).astype('uint8') 
    for node in vertex:
        x, y = node[0], node[1]
        _img = cv2.circle(_img, (int(x), int(y)), 10, (255,0,0), 3)
    #print(_img)
    cv2.imwrite('./{}1.jpg'.format('result'), _img)
        
if __name__=='__main__':
    imgpath = args.imagepath

    if args.mode =='train':
        main_train(imgpath)
    elif args.mode =='test':
        main_test(imgpath)
    elif args.mode == 'predict':
        predict(imgpath)
