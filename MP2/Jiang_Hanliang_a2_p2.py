import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
import imageio
import torch
from skimage.transform import resize
from scipy.ndimage import gaussian_laplace



def part2(path, operation):
    input = cv2.imread(path)
    if(operation == "left"):
        input = input[:, int(input.shape[1] * 0.2):]
    elif(operation == "right"):
        input = input[:, :-int(input.shape[1] * 0.2)]
    elif(operation == "counterclockwise"):
        input = cv2.rotate(input, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif(operation == "clockwise"):
        input = cv2.rotate(input, cv2.ROTATE_90_CLOCKWISE)
    elif(operation == "enlarge"):
        input = cv2.resize(input, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        input = input[input.shape[0]//4:input.shape[0]*3//4,input.shape[1]//4:input.shape[1]*3//4]


    init_scale = 2
    levels = 10
    k = 1.4
    select_parameter = 0.1 
    window = 10       

    original_image = input
    gray = np.float32(cv2.cvtColor(input, cv2.COLOR_BGR2GRAY))/255
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.06)
    pyramid = []
    divide_scale = init_scale
    for i in range(levels):
        pyramid.append(gray)
        gray = resize(gray, ((int)(gray.shape[0]/divide_scale), (int)(gray.shape[1]/divide_scale)), preserve_range=True)
        divide_scale = k

    
    dx = cv2.Sobel(harris, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(harris, cv2.CV_64F, 0, 1, ksize=3)
    orientation = np.arctan2(dy, dx) 
    mask = np.zeros_like(orientation, dtype=bool)
    mask[harris> select_parameter * harris.max()] = True
    orientation = np.where(mask, orientation, 0)
    for i in range(harris.shape[0]-window+1):
        for j in range(harris.shape[1]-window+1):
            sum = 0
            count = 0
            for m in range(window):
                  for n in range(window):
                    if(orientation[i+m][j+n]!=0):
                        sum = sum + orientation[i+m][j+n]
            if(count!=0):
                sum = sum/count
                for m in range(window):
                    for n in range(window):
                        if(orientation[i+m][j+n]!=0):
                            orientation[i+m][j+n] = sum
    count = 0
    for i in range(harris.shape[0]):
        for j in range(harris.shape[1]):
            if(harris[i][j]>select_parameter * harris.max()):
                count = count + 1
                best_response = best_scale = -np.inf
                for level in range(levels):
                    new_i = int(i / (init_scale * (k ** level)))
                    new_j = int(j / (init_scale * (k ** level)))
                    response = np.abs(gaussian_laplace(pyramid[level], init_scale * (k ** level))[new_i][new_j])
                    if(response > best_response):
                        best_response = response
                        best_scale = level
                radius = init_scale * (k ** best_scale)
                cv2.drawMarker(original_image, (j, i), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
                cv2.circle(original_image, (j, i), int(radius), (0, 255, 0), thickness=1)
                cv2.putText(original_image, f"{int(radius)}", (j, i+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                arrow_tip = (int(j + 20 * np.cos(orientation[i][j])), int(i + 20 * np.sin(orientation[i][j])))
                cv2.arrowedLine(original_image, (j, i), arrow_tip, (255, 0, 0), thickness=2, tipLength=0.05)
                # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                # plt.title(str(i)+" & "+str(j))
                # plt.show()
                # print("i = "+str(i)+", j = "+str(j)+", orientation = "+str(orientation[i][j]))

    # print("count = "+str(count))
    # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    cv2.imwrite(path+"__"+operation+'.jpg', original_image)
    return






operation = ["output", "left", "right", "clockwise", "counterclockwise", "enlarge"]
for i in range(5, 9):
    for j in operation:
        start_time = time.time()
        part2(str(i)+".jpg", j)
        end_time = time.time()
        total_time = end_time - start_time
        print(total_time)