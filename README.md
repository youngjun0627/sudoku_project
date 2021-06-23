# sudoku-project
Extract and solve sudoku from image using 2-stage Deep learning model.

## Getting Started
How to use
```    
git clone https://github.com/youngjun0627/sudoku_project.git
cd ~~~~~
```

## Environment
- Ubuntu 20.04
- cuda version 11.2 (RTX2080)
- Python 3.8.8

## Requirements
- torch 1.8.1
- tqdm 4.59.0
- scikit-learn 0.22
- pillow 8.1.2
- numpy 1.19.2
- matplotlib 3.3.4
- albumentations 1.0.0


## File description
- 
- 
- 

## Procedure
 > 1. Detect Keypoint.
 > 2. Perform perspective transform om image.
 > 3. Recognize digits of sudoku.
 > 4. Solve sudoku.

## Run the program
1. Generate dataset
Change background path.
Change save path. (image, mask, annotation saving path)
```  
vi generate_sudoku_images.py
``` 
Run generate_sudoku_images.py
```  
python3 generate_sudoku_images.py
```  
최종 사용하루 csv파일을 만듬~
```  
python3 utils.py
```  

2. Train keypoint
```  
cd sudoku_keypoint
python3 main.py
```  

3. Train number_recognition 
```  
cd sudoku_number_recognition 
python3 main.py
```  

4. test
```  
cd Sudoku
python3 main.py --imagepath (your image path)
```  




