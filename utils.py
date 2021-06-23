import csv
import cv2
import numpy as np
import os
from config import params

def imagProc(path,image_name):
    img = cv2.imread(path,cv2.IMREAD_COLOR)         

    rows = img.shape[0]
    cols = img.shape[1]

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                           
    img_gray_copy = img_gray.copy()
    img_gray = cv2.bilateralFilter(img_gray,5,150,150)

    th = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)  

    contours,_ = cv2.findContours(th,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    max_idx,max_area = 0, 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])                  
        if (area > max_area):
            max_area = area
            max_idx = i
        else:
            continue
    cnt = contours[max_idx]
    rotateRect = cv2.minAreaRect(cnt)
    vertex = cv2.boxPoints(rotateRect)
    
    _img = np.zeros(img.shape, dtype=np.uint8)
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)

    _img = cv2.fillConvexPoly(_img, vertex.astype('int') ,color=255)

    '''
    plt.imshow(_img)
    plt.axis('off')
    plt.show()
    '''
    return _img
    
    #x,y,w,h = cv2.boundingRect(cnt)
    '''

    _img = copy.deepcopy(img)
    _img = cv2.resize(_img,(224,224))
    for node in vertex:
        height, width,_ = img.shape
        node[0] = node[0]*224/width
        node[1] = node[1]*224/height
        _img = cv2.circle(_img, tuple(map(int,node)),10, (255,0,0),3)
    plt.imshow(_img)
    plt.axis('off')
    plt.show()
    '''
    #_img = cv2.rectangle(_img, (x,y), (x+w,y+h),(255,0,0),5)
    '''

    if 'save' not in os.listdir('.'):
        os.mkdir('save')
        save_path = os.path.join('./save', image_name)
        cv2.imwrite(save_path, _img)

    plt.imshow(_img)
    plt.axis('off')
    plt.show()

    '''
    #return vertex

def sampling(image_rootpath, csv_path):
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['image_name','label','x1','y1','x2','y2'])
        for i, image_name in enumerate(os.listdir(image_rootpath)):
            if image_name.endswith('mask.jpg'):
                continue
            image_path = os.path.join(image_rootpath, image_name)
            img = np.array(cv2.imread(image_path))
            '''
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            '''
        #[x1,y1],[x2,y2],[x3,y3],[x4,y4] = imagProc(image_path,image_name)
        #wr.writerow([image_name,x1,y1,x2,y2,x3,y3,x4,y4])
        _img = imagProc(image_path,image_name)
        cv2.imwrite(os.path.join('./train',image_name + '_mask.jpg'),_img)

        if i%100==0:
            print(i)

def train_validation_test_split(dataset_path, train_csv_path, validation_csv_path, test_csv_path, split_ratio, k=1):
    train_csv = csv.writer(open(train_csv_path, 'w', encoding='utf-8-sig', newline=''))
    validation_csv = csv.writer(open(validation_csv_path, 'w', encoding='utf-8-sig', newline=''))
    test_csv = csv.writer(open(test_csv_path, 'w', encoding='utf-8-sig', newline=''))
    #X_train = []
    #Y_train = []
    index = 1
    images_folder = os.path.join(dataset_path, 'sudoku_images')
    masks_folder = os.path.join(dataset_path, 'sudoku_masks')
    masks2_folder = os.path.join(dataset_path, 'sudoku_masks2')
    annotations_folder = os.path.join(dataset_path, 'sudoku_annotations')
    test_valid_index = 1
    for imagename in os.listdir(images_folder): # images folder & masks folder
        csvname = imagename[:-4] + '.csv'
        image_path = os.path.join(images_folder, imagename)
        mask_path = os.path.join(masks_folder, imagename)
        mask2_path = os.path.join(masks2_folder, imagename)
        annotation_path = os.path.join(annotations_folder, csvname)
        f = open(annotation_path, 'r', encoding='utf-8-sig')
        rdr = csv.reader(f)
        for line in rdr:
            sudoku = list_to_num(line[0],'sudoku')
            solution = list_to_num(line[1], 'solution')
            keypoint = list_to_num(line[2],'keypoint')
        if len(keypoint)!=8:
            print(csvname)
            continue

        if (index+k)%(int(1/split_ratio))!=0:
            train_csv.writerow([image_path, mask_path, mask2_path, sudoku, solution, keypoint])
        else:
            if test_valid_index%10!=0:
                validation_csv.writerow([image_path, mask_path, mask2_path, sudoku, solution,keypoint])
            else:
                test_csv.writerow([image_path, mask_path, mask2_path, sudoku, solution,keypoint])
            test_valid_index+=1
        index+=1

def list_to_num(_list, mode):
    _list =  _list.replace('[','')
    _list =  _list.replace(']','')
    _list =  _list.replace(')','')
    _list =  _list.replace('(','')
    if mode=='sudoku' or mode=='solution':
        _list =  list(map(int,_list.split(',')))
    elif mode=='keypoint':
        _list =  list(map(float,_list.split(',')))
    return _list

def calculate_mask2_weights(label_path='./train.csv'):
    '''
    f = open(label_path,'r', encoding='utf-8-sig')
    rdr = csv.reader(f)
    dic = np.array([x for x in range(82)])
    for n,line in enumerate(rdr):
        mask2_path = line[2]
        img = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        for i,number in enumerate(dic):
            dic[i] += (img==i).astype('int').sum()

    
    result = np.array([n*600*600 for _ in range(82)])
    result = result / (dic*82)

    return result
    '''
    return [4.62525631, 4.63220993, 4.63623119, 4.63813273, 4.63734552, 4.63742598,
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

def calculate_sudoku_weights(label_path='/home/guest0/sudoku/sudoku_key_point/train.csv'):
    '''
    f = open(label_path,'r', encoding='utf-8-sig')
    rdr = csv.reader(f)
    dic = np.array([x for x in range(10)])
    for n,line in enumerate(rdr):
        sudoku = line[3]
        sudoku = list(map(int,[x for x in sudoku if x.isdigit()]))
        for su in sudoku:
            dic[su]+=1
    dic = np.array(dic)
    dic = dic.sum()/(dic*10)
    return dic
    '''
    return [0.18053549, 2.00636998, 2.01437749, 2.06482738, 2.05488307, 2.00132495,
     2.01830684, 2.00403835, 1.99313268, 2.00287456]



if __name__=='__main__':


    
    train_csv_path = os.path.join(params['csv_savepath'], 'train.csv')
    validation_csv_path = os.path.join(params['csv_savepath'], 'validation.csv')
    test_csv_path = os.path.join(params['csv_savepath'],'test.csv')
    split_ratio = 0.3
    k=0
    train_validation_test_split(params['dataset_savepath'], train_csv_path, validation_csv_path, test_csv_path, split_ratio,k)
    
    #print(calculate_mask2_weights(label_path='/home/guest0/sudoku/sudoku_segmentation/train.csv'))
