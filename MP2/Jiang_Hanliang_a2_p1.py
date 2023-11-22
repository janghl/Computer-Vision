import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
import imageio
import torch
from skimage.transform import resize
from scipy.ndimage import gaussian_laplace


def part1():
    path='01047u.tif'
    preprocess = True
    visual = 2


    input = imageio.imread(path)     #got this line from chatgpt because PIL cannot read tiff images
    input = (input / np.max(input) * 255).astype(np.uint8)
    row0,column0=np.shape(input)
    input=np.array(input)[:(row0//3*3)-row0,:]
    layer1,layer2,layer3=input.reshape(3,row0//3,column0)
    row0=row0//3
    element1 = torch.from_numpy(layer1).clone().detach().numpy()
    element2 = torch.from_numpy(layer2).clone().detach().numpy()
    element3 = torch.from_numpy(layer3).clone().detach().numpy()

    start_time = time.time()

    #FFT
    if(preprocess == True):
        sharpening_kernel = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        layer1 = cv2.filter2D(layer1, -1, sharpening_kernel)
        layer2 = cv2.filter2D(layer2, -1, sharpening_kernel)
        layer3 = cv2.filter2D(layer3, -1, sharpening_kernel)
    FT1 = np.fft.fft2(layer1)
    FT1 = np.fft.fftshift(FT1)
    FT2 = np.fft.fft2(layer2)
    FT2 = np.fft.fftshift(FT2)
    product2 = FT1 * np.conjugate(FT2)
    inverse2 = np.abs(np.fft.ifft2(product2))
    if(visual==2):
        plt.imshow(inverse2)
        if(preprocess == True):
            plt.title("G to R alignment with preprocessing")
        else:
            plt.title("G to R alignment without preprocessing")
        plt.show()
    offset2 = np.unravel_index(np.argmax(inverse2), inverse2.shape)
    offset_x2 = offset2[1]
    offset_y2 = offset2[0]
    if(offset_x2>column0//2):
        offset_x2 = column0 - offset_x2
    if(offset_y2>row0//2):
        offset_y2 = row0 - offset_y2
    print("offset_x2: "+str(offset_x2))
    print("offset_y2: "+str(offset_y2))


    FT3 = np.fft.fft2(layer3)
    FT3 = np.fft.fftshift(FT3)
    product3 = FT1 * np.conjugate(FT3)
    inverse3 = np.abs(np.fft.ifft2(product3))
    if(visual==3):
        plt.imshow(inverse3)
        if(preprocess == True):
            plt.title("B to R alignment with preprocessing")
        else:
            plt.title("B to R alignment without preprocessing")
        plt.show()
    offset3 = np.unravel_index(np.argmax(inverse3), inverse3.shape)
    offset_x3 = offset3[1]
    offset_y3 = offset3[0]
    if(offset_x3>column0//2):
        offset_x3 = column0 - offset_x3
    if(offset_y3>row0//2):
        offset_y3 = row0 - offset_y3
    print("offset_x3: "+str(offset_x3))
    print("offset_y3: "+str(offset_y3))

    result_x2 = offset_x2
    result_y2 = offset_y2
    result_x3 = offset_x3
    result_y3 = offset_y3


    translation2=np.zeros(np.shape(element2))
    if(result_y2>=0 and result_x2>=0):
        translation2[result_y2:,result_x2:]=element2[:row0-result_y2,:column0-result_x2]
    elif(result_y2>=0 and result_x2<=0):
        translation2[result_y2:,:-result_x2]=element2[:row0-result_y2,result_x2+column0:]
    elif(result_y2<=0 and result_x2>=0):
        translation2[:-result_y2,result_x2:]=element2[result_y2+row0:,:column0-result_x2]
    else:
        translation2[:-result_y2,:-result_x2]=element2[result_y2+row0:,result_x2+column0:]


    translation3=np.zeros(np.shape(element3))
    if(result_y3>=0 and result_x3>=0):
        translation3[result_y3:,result_x3:]=element3[:row0-result_y3,:column0-result_x3]
    elif(result_y3>=0 and result_x3<=0):
        translation3[result_y3:,:-result_x3]=element3[:row0-result_y3,result_x3+column0:]
    elif(result_y3<=0 and result_x3>=0):
        translation3[:-result_y3,result_x3:]=element3[result_y3+row0:,:column0-result_x3]
    else:
        translation3[:-result_y3,:-result_x3]=element3[result_y3+row0:,result_x3+column0:]

    result=np.zeros([row0,column0,3])
    result[:,:,0]=element1
    result[:,:,1]=translation2
    result[:,:,2]=translation3
    cv2.imwrite(path+'_output.jpg',result)
    return

part1()