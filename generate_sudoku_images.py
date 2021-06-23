'''
reference : https://github.com/RutledgePaulV/sudoku-generator
'''

import sys
from Sudoku.Generator import *
import cv2
import numpy as np
import albumentations
import matplotlib.pyplot as plt
import os
import copy
from tqdm import tqdm
import csv
import random
from multiprocessing import Process
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

def generate_sudoku(difficulty_string, base = 'base.txt'):
# setting difficulties and their cutoffs for each solve method
    difficulties = {
      'easy': (35, 0), 
      'medium': (81, 5), 
      'hard': (81, 10), 
      'extreme': (81, 15)
      }

    # getting desired difficulty from command line
    difficulty = difficulties[difficulty_string]

    # constructing generator object from puzzle file (space delimited columns, line delimited rows)
    gen = Generator(base)

    # applying 100 random transformations to puzzle
    gen.randomize(100)

    # getting a copy before slots are removed
    initial = gen.board.copy()

    # applying logical reduction with corresponding difficulty cutoff
    gen.reduce_via_logical(difficulty[0])

    # catching zero case
    if difficulty[1] != 0:
        # applying random reduction with corresponding difficulty cutoff
        gen.reduce_via_random(difficulty[1])


    # getting copy after reductions are completed
    final = gen.board.copy()

    # printing out complete board (solution)
    #print("The initial board before removals was: \r\n\r\n{0}".format(initial))

    # printing out board after reduction
    #print("The generated board after removals was: \r\n\r\n{0}".format(final))
  
    solution = np.array(list(map(int,[x for x in str(initial) if x.isdigit()]))).reshape(9,9).tolist()
  
    sudoku = []
    for x in str(final):
        if x.isdigit():
            sudoku.append(int(x))
        elif x == '_':
            sudoku.append(0)
    sudoku = np.array(sudoku).reshape(9,9).tolist()
    return sudoku, solution

import cv2
import numpy as np
import albumentations
import matplotlib.pyplot as plt
import os
import copy
from tqdm import tqdm
import csv
import random

#np.set_printoptions(threshold=600*600*3,linewidth=np.inf)
def generate_sudoku_image(imagename, original_img_path, sudoku,solution):
    image = np.zeros((174+9*28+174, 174+9*28+174, 3), dtype=np.uint8) # 이미지 넣기
    image[:,:,:] = [255,0,255]
    image2 = np.zeros((174+9*28+174, 174+9*28+174), dtype=np.uint8) # 이미지 넣기
    image2[:,:] = 81
    image3 = np.zeros((600,600), dtype=np.uint8)
    image3[174:426, 174:426]=255
    background = np.random.randint(0,30,size=1)
    for i in range(9):
        for j in range(9):
            image2[174+i*28:174+(i*28)+28,174+j*28:174+(j*28)+28] = i*9+j
            if sudoku[i][j] != 0:
                image[174+i*28:174+(i*28)+28,174+j*28:174+(j*28)+28,:] = read_mnist2(sudoku[i][j], background)
            else:
                image[174+i*28:174+(i*28)+28,174+j*28:174+(j*28)+28,:] = background
    
    for i in range(9):
        image[174+i*28:174+(i*28)+1,174+0*28:174+(9*28)+1,:] = np.array([np.random.randint(170,255, size=9*28+1) for _ in range(3)]).reshape(1,-1,3)
    image[174+9*28-1,174+0*28:174+(9*28)+1,:] = np.array([np.random.randint(170,255,size=9*28+1) for _ in range(3)]).reshape(-1,3)
    for i in range(9):
        h_s = np.random.randint(1,3,size=1)
        h_e = np.random.randint(1,3,size=1)
        w_s = np.random.randint(1,3,size=1)
        w_e = np.random.randint(1,3,size=1)
        image[174+0*28:174+(9*28)+1,174+i*28:174+(i*28)+1,:] = np.array([np.random.randint(170,255,size=9*28+1) for _ in range(3)]).reshape(-1,1,3)
    image[174+0*28:174+(9*28)+1,174+9*28-1,:] = np.array([np.random.randint(170,255,size=9*28+1) for _ in range(3)]).reshape(-1,3)


    sudoku_image = copy.deepcopy(255-image)
    #M = cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2), 45, 1)
    #image = cv2.warpAffine(image, M, (500,500), borderValue=(255,255,255))
    transform = create_transform()

    original_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    original_img = cv2.resize(original_img, (600,600),interpolation=cv2.INTER_CUBIC)
    key_point = [(174,174),(426,174),(174,426),(426,426)]
    while True:
        image=copy.deepcopy(sudoku_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (50, 150, 0), (70, 255, 255))
        cv2.copyTo(original_img, mask, image)
    
        annotation = copy.deepcopy(image3)
        annotation2 = np.array(image2).astype('uint8')
        tr = transform(image = image, mask=annotation, mask2=annotation2, keypoints = key_point)
        image = tr['image']
        mask = tr['mask']
        mask2 = tr['mask2']
        key = tr['keypoints']
        if check_boarder(mask) and len(key)==4:
            break
    '''
    plt.imshow(image)
    plt.show()
    plt.imshow(mask)
    plt.show()
    '''

    save_images_path = '/mnt/data/guest0/sudoku_dataset/sudoku_images'
    save_masks_path = '/mnt/data/guest0/sudoku_dataset/sudoku_masks'
    save_masks2_path = '/mnt/data/guest0/sudoku_dataset/sudoku_masks2'
    save_annotations_path ='/mnt/data/guest0/sudoku_dataset/sudoku_annotations'

    save_images_path
    if not os.path.exists(save_images_path):
        os.mkdir(save_images_path)
    if not os.path.exists(save_masks_path):
        os.mkdir(save_masks_path)
    if not os.path.exists(save_masks2_path):
        os.mkdir(save_masks2_path)
    if not os.path.exists(save_annotations_path):
        os.mkdir(save_annotations_path)

    with open(os.path.join(save_annotations_path,imagename[:-4] + '.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([sudoku, solution, key])
        
    cv2.imwrite(os.path.join(save_images_path,imagename), image)
    cv2.imwrite(os.path.join(save_masks_path,imagename), mask)
    cv2.imwrite(os.path.join(save_masks2_path, imagename), mask2)

    #return image

def check_boarder(mask):
    
    h,w = mask.shape
    for i in range(h):
        if (mask[i,0]==255).all():
            return False
        if (mask[i,w-1]==255).all():
            return False
    for i in range(w):
        if (mask[0,i]==255).all():
            return False
        if (mask[h-1,0]==255).all():
            return False
    return True

def create_transform():
    translist = []

    translist += albumentations.OneOf([
        albumentations.ShiftScaleRotate(interpolation = 2, shift_limit=0.1, scale_limit = 0.5, border_mode = 3,p=0.7),
            albumentations.ShiftScaleRotate(interpolation = 2, p=0.7),
            albumentations.Perspective(scale = (0.05,0.1), interpolation = 2, keep_size=True, p=0.5)],p=1)
        
    translist += [albumentations.Blur(blur_limit=3)]
    translist += [albumentations.GaussianBlur(blur_limit=(3,5))]
    i#translist += [albumentations.MedianBlur()]
    translist += [albumentations.MotionBlur(blur_limit=(3,5))]
    translist += [albumentations.ColorJitter()]
    translist = albumentations.Compose(translist, additional_targets = {'mask2':'mask'}, keypoint_params=albumentations.KeypointParams(format='xy'))
    return translist

def read_mnist(label, background):
    root = '/mnt/data/guest0/sudoku_dataset/mnist_png/training'
    number = np.random.randint(0,5000,1)[0]
    
    imagename = os.listdir(os.path.join(root,str(label)))[number]
  
    img = cv2.imread(os.path.join(root,str(label),imagename))
    img[img!=0] = np.random.randint(235,255,size=(img!=0).sum())
    img[img==0] = background
    
    translist = []
    translist += [albumentations.Blur(blur_limit=3)]
    translist += [albumentations.GaussianBlur(blur_limit=(3,5))]
    translist += [albumentations.MotionBlur(blur_limit=(3,5))]
    translist = albumentations.Compose(translist)
    img = translist(image = img)['image']
  
    return img


def read_mnist2(label,background):
    img = Image.fromarray(np.zeros((28,28,3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./arial.ttf", 20)
    draw.text((8, 4), str(label),(255,255,255),font=font)
    img = np.array(img)
    img[img!=0] = np.random.randint(245,255,size=(img!=0).sum())
    img[img==0] = background
    '''
    translist = []
    translist += [albumentations.Blur(blur_limit=3)]
    translist += [albumentations.GaussianBlur(blur_limit=(3,5))]
    translist += [albumentations.MedianBlur()]
    translist += [albumentations.MotionBlur()]
    translist += [albumentations.ColorJitter()]
    translist = albumentations.Compose(translist)
    img = translist(image = img)['image']
    '''
    return img



def generate(diff, datapath, out_index):
    for idx, img_path in enumerate(os.listdir(datapath)):
        print((744*out_index+idx))
        if img_path.endswith('.png') or img_path.endswith('.jpg'):
            img_path = os.path.join(datapath, img_path)
            number = np.random.randint(0,4,1)[0]
    
            sudoku,solution = generate_sudoku(diff[number])
            imagename = str(744*out_index+idx) + '.png'
            img = generate_sudoku_image(imagename, img_path, sudoku, solution)

  


if __name__=='__main__':
    diff = ['easy', 'medium', 'hard', 'extreme']
    datapath = '/mnt/data/guest0/sudoku_dataset/newspapers/data'
    
    procs = []
    for out_index in range(10):
        proc = Process(target=generate, args=(diff, datapath, out_index))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
