import numpy as np
from PIL import Image
import matplotlib as plt
import cv2
import time
import imageio
start_time = time.time()
def NCC(array1,array2):
    size=np.size(array1)
    array1=array1-np.mean(array1)
    array2=array2-np.mean(array2)
    array1=array1/np.linalg.norm(array1, ord=2)
    array2=array2/np.linalg.norm(array2, ord=2)
    return np.sum(np.dot(array1,array2))

path='01657u.tif'
multiscale=True
# input=Image.open(path)
# input=np.array(input)
input = imageio.imread(path)     #got this line from chatgpt because PIL cannot read tiff images
input = (input / np.max(input) * 255).astype(np.uint8)
# cv2.imwrite('test.jpg',input)
#print(np.shape(input))
row0,column0=np.shape(input)
input=np.array(input)[:(row0//3*3)-row0,:]
element1,element2,element3=input.reshape(3,row0//3,column0)
row0=row0//3
# cv2.imshow('output',element1)
layer1,layer2,layer3=element1,element2,element3
result2=0
result_x2=0
result_y2=0
result3=0
result_x3=0
result_y3=0
if(multiscale==True):
    for i in range(3):
        layer1=layer1[::2,::2]
        layer2=layer2[::2,::2]
        layer3=layer3[::2,::2]
    row,column=np.shape(layer1)
window=15
midrow=row//2
midcolumn=column//2
array1=element1[midrow-window:midrow+window+1,midcolumn-window:midcolumn+window+1].flatten()
#assume layer2 moves x to the right and y downwards
for y in range(window-row+midrow+1,midrow-window+1):
    for x in range(window-column+midcolumn+1,midcolumn-window+1):
        #print(midrow-y-window,' ',midrow-y+window,' ',midcolumn-x-window,' ',midcolumn-x+window,' ',result)
        array2=element2[midrow-y-window:midrow-y+window+1,midcolumn-x-window:midcolumn-x+window+1].flatten()
        if(NCC(array1,array2)>result2):
            result_x2=x
            result_y2=y
            result2=NCC(array1,array2)
print('x=',result_x2,'y=',result_y2,'result=',result2)
# layer3
for z in range(window-row+midrow+1,midrow-window+1):
    for w in range(window-column+midcolumn+1,midcolumn-window+1):
        array3=element3[midrow-z-window:midrow-z+window+1,midcolumn-w-window:midcolumn-w+window+1].flatten()
        if(NCC(array1,array3)>result3):
            result_x3=w
            result_y3=z
            result3=NCC(array1,array3)
print('x=',result_x3,'y=',result_y3,'result=',result3)


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

# slide only in the center
# # for z in range(-window,window+1):
# #     for w in range(-window,window+1):
# #         array3=element3[midrow-z-window:midrow-z+window+1,midcolumn-w-window:midcolumn-w+window+1].flatten()
# #         if(NCC(array1,array3)>result3):
# #             result_x3=w
# #             result_y3=z
# #             result3=NCC(array1,array3)
# # print('x=',result_x3,'y=',result_y3,'result=',result3)
# # translation3=np.zeros(np.shape(element3))
# # if(result_y3>=0 and result_x3>=0):
# #     translation3[result_y3:,result_x3:]=element3[:row-result_y3,:column-result_x3]
# # elif(result_y3>=0 and result_x3<=0):
# #     translation3[result_y3:,:-result_x3]=element3[:row-result_y3,result_x3+column:]
# # elif(result_y3<=0 and result_x3>=0):
# #     translation3[:-result_y3,result_x3:]=element3[result_y3+row:,:column-result_x3]
# # else:
# #     translation3[:-result_y3,:-result_x3]=element3[result_y3+row:,result_x3+column:]

result=np.zeros([row0,column0,3])
result[:,:,0]=element1
result[:,:,1]=translation2
result[:,:,2]=translation3
cv2.imwrite(path+'_output.jpg',result)
end_time = time.time()
total_time = end_time - start_time
print(total_time)
