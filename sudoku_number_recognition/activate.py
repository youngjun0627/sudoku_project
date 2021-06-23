import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import csv

def train(model, criterion, optimizer, train_dataloader, device):
    model.train()
    train_loss = 0.0
    score = 0
    for i, (input, sudoku) in enumerate(tqdm(train_dataloader)):
    #for input, mask, mask2, keypoint in train_dataloader:
        input = input.to(device)
        sudoku = sudoku.to(device)
        
        output = model(input)

        loss = criterion(output,sudoku)
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(),0.1)
        optimizer.step()
        train_loss += loss.data.item()

        with torch.no_grad():
            score += (torch.argmax(output,dim=1)==sudoku).long().sum() / len(sudoku.reshape(-1))
    train_loss /= len(train_dataloader)
    score /= len(train_dataloader)
    return train_loss, score


def valid(model, criterion, valid_dataloader, device):
    model.eval()
    valid_loss = 0.0
    score = 0
    with torch.no_grad():
        for i, (input, sudoku) in enumerate(tqdm(valid_dataloader)):
            input = input.to(device)
            sudoku = sudoku.to(device)
        
            output = model(input)

            loss = criterion(output,sudoku)

            valid_loss += loss.data.item()
            
            score += (torch.argmax(output,dim=1)==sudoku).long().sum() / len(sudoku.reshape(-1))

        valid_loss /= len(valid_dataloader)
        score /= len(valid_dataloader)
    return valid_loss, score

def test(model, test_dataloader, device):
    model.eval()
    score = 0
    with torch.no_grad():
        for i, (input,sudoku) in enumerate(tqdm(test_dataloader)):
            
            input = input.to(device)
            sudoku = sudoku.to(device)
            output = model(input)
            
            score += (torch.argmax(output,dim=1)==sudoku).long().sum().item() / len(sudoku.reshape(-1))
        score /= len(test_dataloader)
    return score

            
