# sudoku-project
Extract and solve sudoku from image using 2-stage Deep learning model.

## Getting Started
How to install
```    
git clone https://github.com/youngjun0627/sudoku_project.git
cd sudoku
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

## Procedure
 > 1. Detect Keypoint.
 > 2. Perform perspective transform om image.
 > 3. Recognize digits of sudoku.
 > 4. Solve sudoku.

## Run the program
### Generate dataset
Change background path and save path(image, mask, annotation saving path).
```  
vi generate_sudoku_images.py
``` 
Run generate_sudoku_images.py
```  
python3 generate_sudoku_images.py
```  
Generate final csv files.
```  
python3 utils.py
```  

### Train keypoint
```  
cd sudoku_keypoint
python3 main.py
```  

### Train number_recognition 
```  
cd sudoku_number_recognition 
python3 main.py
```  

### Test
```  
cd Sudoku
python3 main.py --imagepath (your image path)
```  

## Working

#### Generated image of sudoku-
![Generated image of sudoku](img 경로~)

#### Detect corners of sudoku-
![Corners of sudoku](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/contour.jpg)

#### Image of Sudoku after keypoint detection and perspective transform-
![Transformed image of sudoku](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/threshold.jpg)

#### Recognize digits of sudoku-
![Detection sudoku image](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/final.jpg)

#### Solve sudoku-
![Solved sudoku image](https://github.com/aakashjhawar/SolveSudoku/blob/master/images/final.jpg)



