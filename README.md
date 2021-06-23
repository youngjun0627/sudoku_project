# sudoku-project
Extract and solve sudoku from image using 2-stage Deep learning model.

## Getting Started
How to install
```    
git clone https://github.com/youngjun0627/sudoku_project.git
cd sudoku
```
Change to your paths.
```   
vi config.py
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
![Generated image of sudoku](https://user-images.githubusercontent.com/68416187/123059168-76be9c00-d444-11eb-9961-bac246d98615.png)

#### Mask images of sudoku keypoint detection(box, cells, keypoints)-
<img src="https://user-images.githubusercontent.com/68416187/123059541-dae16000-d444-11eb-9200-1441f5f48b2f.PNG"  width="250" height="250">  <img src="https://user-images.githubusercontent.com/68416187/123059679-fa788880-d444-11eb-94c8-90dfd05c6485.PNG"  width="250" height="250">  <img src="https://user-images.githubusercontent.com/68416187/123059714-03695a00-d445-11eb-8715-8e8007c30e1e.PNG"  width="250" height="250">

#### 
https://github.com/youngjun0627/sudoku_project/blob/main/result1.jpg
#### Image of sudoku after keypoint detection and perspective transform-
![Transformed image of sudoku](https://github.com/youngjun0627/sudoku_project/blob/main/result.jpg)

#### Recognize digits of sudoku-
![Detection sudoku image](https://github.com/youngjun0627/sudoku_project/blob/main/detection%20sudoku.jpg)

#### Solve sudoku-
![Solved sudoku image](https://github.com/youngjun0627/sudoku_project/blob/main/solution.jpg)



