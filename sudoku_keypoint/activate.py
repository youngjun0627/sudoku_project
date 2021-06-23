import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_similarity_score
from metric import dice_coeff
import cv2
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import csv

def train(model, seg_criterion1, seg_criterion2, key_criterion, optimizer, train_dataloader, device):
    model.train()
    train_loss = 0.0
    seg_score1 = 0
    seg_score2 = 0
    score = 0
    for i, (input, mask, mask2, keypoint) in enumerate(tqdm(train_dataloader)):
    #for input, mask, mask2, keypoint in train_dataloader:
        input = input.to(device)
        mask = mask.to(device)
        mask2 = mask2.to(device)
        keypoint = keypoint.to(device)
        
        seg1, seg2, key = model(input)

        seg1_loss = seg_criterion1(seg1, mask)
        seg2_loss = seg_criterion2(seg2, mask2)
        key_loss = key_criterion(key, keypoint)
        #print('seg1 : {},  seg2 : {},  key : {}'.format(seg1_loss.item(), seg2_loss.item(), key_loss.item()))
        loss = seg1_loss + seg2_loss + key_loss
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(),0.1)
        optimizer.step()
        train_loss += loss.data.item()

        with torch.no_grad():
            pred = torch.argmax(seg2, axis=1).long()
            seg_score2 += (mask2==pred).long().sum().item() / (mask2.size(0)*mask2.size(1)*mask2.size(2))
            seg_score1 += dice_coeff((torch.sigmoid(seg1)>0.5).float(), mask, device).item()
            for i in range(input.size(0)):
                for _k,k in zip(key[i,:,:,:], keypoint[i,:,:,:]):
                    if abs(((torch.argmax(_k)-torch.argmax(k))).item()) < 20:
                        score+=1
                score/=4
            score/=input.size(0)

    train_loss /= len(train_dataloader)
    seg_score1 /= len(train_dataloader)
    seg_score2 /= len(train_dataloader)
    score /= len(train_dataloader)
    return train_loss, seg_score1, seg_score2, score


def valid(model, seg_criterion1, seg_criterion2, key_criterion, valid_dataloader, device):
    model.eval()
    valid_loss = 0.0
    seg_score1 = 0
    seg_score2 = 0
    score = 0
    with torch.no_grad():
        for i, (input, mask, mask2, keypoint) in enumerate(tqdm(valid_dataloader)):
            input = input.to(device)
            mask = mask.to(device)
            mask2 = mask2.to(device)
            keypoint = keypoint.to(device)
            seg1,seg2,key = model(input)
            
            seg1_loss = seg_criterion1(seg1, mask)
            seg2_loss = seg_criterion2(seg2, mask2)
            key_loss = key_criterion(key, keypoint)

            loss = seg1_loss + seg2_loss + key_loss

            valid_loss += loss.data.item()
            
            pred = torch.argmax(seg2, axis=1).long()
            seg_score2 += (mask2==pred).long().sum().item() / (mask2.size(0)*mask2.size(1)*mask2.size(2))
            seg_score1 += dice_coeff((torch.sigmoid(seg1)>0.5).float(), mask, device).item()
            for i in range(input.size(0)):
                for _k,k in zip(key[i,:,:,:], keypoint[i,:,:,:]):
                    if abs(((torch.argmax(_k)-torch.argmax(k))).item()) < 20:
                        score+=1/4
            score/=input.size(0)
            #score += dice_coeff((torch.sigmoid(key)>0.7).float(), keypoint, device).item() 
            #score += F.l1_loss(key, keypoint).item() 

        valid_loss /= len(valid_dataloader)
        seg_score1 /= len(valid_dataloader)
        seg_score2 /= len(valid_dataloader)
        score /= len(valid_dataloader)
    return valid_loss, seg_score1, seg_score2,  score

def test(model, test_dataloader, device):
    model.eval()
    seg_score1 = 0
    seg_score2 = 0
    score = 0
    with torch.no_grad():
        for i, (input, mask, mask2, keypoint) in enumerate(tqdm(test_dataloader)):
            input = input.to(device)
            mask = mask.to(device)
            mask2 = mask2.to(device)
            keypoint = keypoint.to(device)
            seg1,seg2,key = model(input)
            
            pred = torch.argmax(seg2, axis=1).long()
            seg_score2 += (mask2==pred).long().sum().item() / (mask2.size(0)*mask2.size(1)*mask2.size(2))
            seg_score1 += dice_coeff((torch.sigmoid(seg1)>0.5).float(), mask, device).item()
            for _k,k in zip(key[0,:,:,:], keypoint[0,:,:,:]):
                if abs(((torch.argmax(_k)-torch.argmax(k))).item()) < 20:
                    score+=1/4
            #score += dice_coeff((torch.sigmoid(key)>0.7).float(), keypoint, device).item() 
            
            #score += F.l1_loss(key, keypoint).item() 

        seg_score1 /= len(test_dataloader)
        seg_score2 /= len(test_dataloader)
        score /= len(test_dataloader)
        print(seg_score1, seg_score2, score)
    return seg_score1, seg_score2,  score
    '''
    model.eval()
    with torch.no_grad():
        for i, (input,_) in enumerate(tqdm(test_dataloader)):
            
            input = input.to(device)
            output = model(input)
            
            _img = input.squeeze(0).permute(1,2,0).cpu().numpy().astype('uint8')
            for idx in range(0,len(output),2):
            
                x, y = output[idx], output[idx+1]
                _img = cv2.circle(_img, tuple(int(x), int(y)), 10, (255,0,0), 3)
            cv2.imwrite('/home/guest0/sudoku/sudoku_segmentation/seg_result/{}.jpg'.format('result'), _img)


    '''        
