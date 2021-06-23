import torch
from sudoku_keypoint.model import Keypoint_Net
from sudoku_number_recognition.model import Number_Net
import cv2
import numpy as np
import os
import argparse
import copy
from SolveSudoku import sudoku_solver
from check_sudoku import check_sudoku
import csv
from draw_sudoku import draw_mat
from config import params

parser = argparse.ArgumentParser(description='sudoku')
parser.add_argument('--imagepath', default = '.', type=str)
args = parser.parse_args()

def main(img_path):

    BATCHSIZE = 1
    ### custom ###
    keymodel_path = params['key_model']
    nummodel_path = params['num_model']
    device = torch.device('cuda:{}'.format(params['gpu']))

    assert os.path.exists(keymodel_path) or os.path.exists(nummodel_path)


    #model = UNetWithResnet50Encoder().to(device)
    model1 = Keypoint_Net()
    model1.load_state_dict(torch.load(keymodel_path))
    model1 = model1.to(device)
    model2 = Number_Net().to(device)
    model2.load_state_dict(torch.load(nummodel_path))
    model2 = model2.to(device)

    model1.eval()
    model2.eval()
    original_size = 600
    scale = 0.45
    
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(int(original_size*scale), int(original_size*scale)), cv2.INTER_CUBIC)
    input = torch.tensor(img, dtype=torch.float).permute(2,0,1).unsqueeze(0).to(device)
    input /= 255.

    _,_,output = model1(input)

    #print(output)
    output = (output.squeeze(0).cpu().detach())
    #cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}.jpg'.format('result'), _img)
    vertex = []
    for key in output:
        key = key.reshape(-1)
        index = torch.argmax(key)
        vertex.append([index%270, index//270])
   
    _img = copy.deepcopy(img)
    for node in vertex:
        x, y = node[0], node[1]
        _img = cv2.circle(_img, (int(x), int(y)), 10, (255,0,0), 3)
    #cv2.imwrite('./{}.jpg'.format('result1'), _img)
    

    size = original_size * scale
    s_point =  np.array([vertex[0], vertex[1], vertex[3], vertex[2]], dtype=np.float32)
    e_point =  np.array([[0,0],[size,0], [size,size], [0,size]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(s_point, e_point)
    img = cv2.warpPerspective(img, matrix, (int(size),int(size)))
    #cv2.imwrite('./sudoku/{}.jpg'.format('result'), img)
    input = torch.tensor(img, dtype=torch.float).permute(2,0,1).unsqueeze(0).to(device)
    input /= 255.
    sudoku = model2(input)
    sudoku = torch.argmax(sudoku,dim=1).squeeze(0).cpu().numpy().tolist()
    print('original')
    print_sudoku(sudoku)
    _img = draw_mat(sudoku)
    cv2.imwrite('./{}.jpg'.format('detection sudoku'), _img)
    print('-------------------------')
    if check_sudoku(sudoku):    
        print('solution')
        _img = draw_mat(sudoku_solver(sudoku))
        cv2.imwrite('./{}.jpg'.format('solution'), _img)
        print_sudoku(sudoku_solver(sudoku)) 

def test():    

    BATCHSIZE = 1
    ### custom ###
    keymodel_path = params['key_model']
    nummodel_path = params['num_model']
    device = torch.device('cuda:{}'.format(params['gpu']))

    assert os.path.exists(keymodel_path) or os.path.exists(nummodel_path)


    #model = UNetWithResnet50Encoder().to(device)
    model1 = Keypoint_Net()
    model1.load_state_dict(torch.load(keymodel_path))
    model1 = model1.to(device)
    model2 = Number_Net().to(device)
    model2.load_state_dict(torch.load(nummodel_path))
    model2 = model2.to(device)

    model1.eval()
    model2.eval()
    original_size = 600
    scale = 0.45
    
    f = open('./test.csv', 'r', encoding='utf-8-sig')
    rdr = csv.reader(f)
    cnt=0
    total=0
    for line in rdr:
        img_path = line[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img,(int(original_size*scale), int(original_size*scale)), cv2.INTER_CUBIC)
        input = torch.tensor(img, dtype=torch.float).permute(2,0,1).unsqueeze(0).to(device)
        input /= 255.

        _,_,output = model1(input)

        #print(output)
        output = (output.squeeze(0).cpu().detach())
        #cv2.imwrite('/home/guest0/sudoku/sudoku_keypoint/seg_result/{}.jpg'.format('result'), _img)
        vertex = []
        for key in output:
            key = key.reshape(-1)
            index = torch.argmax(key)
            vertex.append([index%270, index//270])
        '''
        _img = copy.deepcopy(img)
        for node in vertex:
            x, y = node[0], node[1]
            _img = cv2.circle(_img, (int(x), int(y)), 10, (255,0,0), 3)
        cv2.imwrite('/home/guest0/sudoku/{}.jpg'.format('result'), _img)
        '''

        size = original_size * scale
        s_point =  np.array([vertex[0], vertex[1], vertex[3], vertex[2]], dtype=np.float32)
        e_point =  np.array([[0,0],[size,0], [size,size], [0,size]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(s_point, e_point)
        img = cv2.warpPerspective(img, matrix, (int(size),int(size)))
        cv2.imwrite('./{}.jpg'.format('result'), img)
        input = torch.tensor(img, dtype=torch.float).permute(2,0,1).unsqueeze(0).to(device)
        input /= 255.
        sudoku = model2(input)
        sudoku = torch.argmax(sudoku,dim=1).squeeze(0).cpu().numpy().tolist()
        #print_sudoku(sudoku)
        #print(check_sudoku(sudoku))
        if check_sudoku(sudoku):   
            #print(sudoku)
            #print(line[3])
            solution = sudoku_solver(sudoku)
            if solution is not None:
                print_sudoku(sudoku)
                print_sudoku(solution) 
                cnt+=1
        total+=1
    print(cnt, total)
def print_sudoku(sudoku):
    for i in range(9):
        _str = ''
        for j in range(9):
            _str += ' ' + str(sudoku[i][j])
        print(_str)

if __name__=='__main__':
    # test
    '''
    test()
    '''
    imgpath = args.imagepath
    main(imgpath)
    
