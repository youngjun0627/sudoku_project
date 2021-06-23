from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

def draw_mat(mat, background=255):
    #img : numpy
    #img_name : str
    #mat : numpy

    W, H = 450, 450
    
    text_color = np.random.randint(0, 100)

    font_size = 30
    font = ImageFont.truetype("arial.ttf", size=font_size)
    
    font_color = (text_color, text_color, text_color)
    
    image = Image.new("RGB", (W, H), (background, background, background))
    draw = ImageDraw.Draw(image)

    cell_size = W // 9
    jinizone_r = cell_size // 3   #지니존~
    jinizone_c = cell_size // 5

    for row in range(9):
        for col in range(9):
            if mat[row][col] == 0:
                n = ''
            else:
                n = str(mat[row][col])

            draw.text((cell_size*col + jinizone_r, cell_size*row + jinizone_c), n, font=font, fill=font_color)

    step_count = 9
    y_start = 0
    y_end = H
    step_size = W // step_count
            
    line_width = np.random.randint(1, 4)  
    #line_color = np.random
    
    for x in range(0, W, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill = font_color, width=line_width)
        
    x_start = 0
    x_end = W
    
    for y in range(0, H, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=font_color, width=line_width)
    
    draw.line(((x_start, H-line_width), (x_end, H-line_width)), fill=font_color, width=line_width)
    draw.line(((W-line_width, y_start), (W-line_width, y_end)), fill=font_color, width=line_width)
    
    del draw
        
    image = image.resize((250, 250))
    #fig = plt.figure()
    #plt.grid(True)
    #plt.imshow(image)
    #imwrite()
    #image.save('{}.jpg'.format(img_name))

    return np.array(image)
