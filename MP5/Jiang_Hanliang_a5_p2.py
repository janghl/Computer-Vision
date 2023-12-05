import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

def disparity_map(window, disparity, method, picture, Bf=3):
          if picture=="moebius":
                    left = cv2.imread('./moebius1.png')
                    right = cv2.imread('./moebius2.png')
          elif picture=="tsukuba":
                    left = cv2.imread('./tsukuba1.jpg')
                    right = cv2.imread('./tsukuba2.jpg')
          left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
          right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
          row, col = left.shape
          if disparity%2==1:
                    buf = np.zeros(disparity)
          else:
                    buf = np.zeros(disparity-1)
          map = np.zeros((row, col))
          for i in range(row-window+1):
                    d = 0
                    for j in range(col-window+1):
                              part1 = left[i:i+window, j:j+window]
                              for k in range(int((-disparity+1)/2), int((disparity+1)/2), 1):
                                        if j+k<0 or j+k>col-window:
                                                  buf[k-int((-disparity+1)/2)]=10000000
                                        else:
                                                  part2 = right[i:i+window, j+k:j+k+window]
                                                  if method=="SSD":
                                                            buf[k-int((-disparity+1)/2)]=np.sum((part1-part2)**2)
                                                  elif method=="SAD":
                                                            buf[k-int((-disparity+1)/2)]=np.sum(abs(part1-part2))
                                                  elif method=="NC":
                                                            buf[k-int((-disparity+1)/2)]=1-np.mean(np.multiply(part1-np.mean(part1), part2-np.mean(part2)))/(np.std(part1)*np.std(part2)+0.0001)
                              d = np.where(np.min(buf)==buf)[0][0]+int((-disparity+1)/2)
                              # if d!=0:
                              #           print(f"d = {d}, i = {i}, j = {j}")
                              # print(f"d = {d}, i = {i}, j = {j}")
                              if not d==0:
                                        map[i][j] = Bf/d
          return map
                                                            


method = "SSD"
# method = "SAD"
# method = "NC"


picture = "moebius"
window = 7
disparity = 150


# picture = "tsukuba"
# disparity = 40
# window = 7
for method in ["NC"]:
          begin = time.time()
          map = disparity_map(window=window, disparity=disparity, method=method, picture=picture)
          end = time.time()
          print(f"total time is {end-begin} s")
          fig, ax = plt.subplots()
          ax.imshow(map, cmap="gray")
          # plt.show()
          print(f"{picture}_w：{window}_d：{disparity}")
          plt.savefig(f"test/{picture}_{method}_w：{window}_d：{disparity}.png")