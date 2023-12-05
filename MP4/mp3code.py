# imports
import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy 
import time
import imageio
import matplotlib as plt
from scipy.spatial.distance import cdist
from pylab import *
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image
from random import sample


##############################################
### Provided code - nothing to change here ###
##############################################
def code(img):
    """
    Harris Corner Detector
    Usage: Call the function harris(filename) for corner detection
    Reference   (Code adapted from):
                http://www.kaij.org/blog/?p=89
                Kai Jiang - Harris Corner Detector in Python
                
    """

    def harris(filename, min_distance = 10, threshold = 0.1):
        """
        filename: Path of image file
        threshold: (optional)Threshold for corner detection
        min_distance : (optional)Minimum number of pixels separating 
        corners and image boundary
        """
        im = np.array(Image.open(filename).convert("L"))
        harrisim = compute_harris_response(im)
        filtered_coords = get_harris_points(harrisim,min_distance, threshold)
        plot_harris_points(im, filtered_coords)

    def gauss_derivative_kernels(size, sizey=None):
        """ returns x and y derivatives of a 2D 
            gauss kernel array for convolutions """
        size = int(size)
        if not sizey:
            sizey = size
        else:
            sizey = int(sizey)
        y, x = mgrid[-size:size+1, -sizey:sizey+1]
        #x and y derivatives of a 2D gaussian with standard dev half of size
        # (ignore scale factor)
        gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
        gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
        return gx,gy

    def gauss_kernel(size, sizey = None):
        """ Returns a normalized 2D gauss kernel array for convolutions """
        size = int(size)
        if not sizey:
            sizey = size
        else:
            sizey = int(sizey)
        x, y = mgrid[-size:size+1, -sizey:sizey+1]
        g = exp(-(x**2/float(size)+y**2/float(sizey)))
        return g / g.sum()

    def compute_harris_response(im):
        """ compute the Harris corner detector response function 
            for each pixel in the image"""
        #derivatives
        gx,gy = gauss_derivative_kernels(3)
        imx = signal.convolve(im,gx, mode='same')
        imy = signal.convolve(im,gy, mode='same')
        #kernel for blurring
        gauss = gauss_kernel(3)
        #compute components of the structure tensor
        Wxx = signal.convolve(imx*imx,gauss, mode='same')
        Wxy = signal.convolve(imx*imy,gauss, mode='same')
        Wyy = signal.convolve(imy*imy,gauss, mode='same')   
        #determinant and trace
        Wdet = Wxx*Wyy - Wxy**2
        Wtr = Wxx + Wyy   
        return Wdet / Wtr

    def get_harris_points(harrisim, min_distance=10, threshold=0.1):
        """ return corners from a Harris response image
            min_distance is the minimum nbr of pixels separating 
            corners and image boundary"""
        #find top corner candidates above a threshold
        corner_threshold = max(harrisim.ravel()) * threshold
        harrisim_t = (harrisim > corner_threshold) * 1    
        #get coordinates of candidates
        candidates = harrisim_t.nonzero()
        coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
        #...and their values
        candidate_values = [harrisim[c[0]][c[1]] for c in coords]    
        #sort candidates
        index = argsort(candidate_values)   
        #store allowed point locations in array
        allowed_locations = zeros(harrisim.shape)
        allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1   
        #select the best points taking min_distance into account
        filtered_coords = []
        for i in index:
            if allowed_locations[coords[i][0]][coords[i][1]] == 1:
                filtered_coords.append(coords[i])
                allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),
                    (coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0               
        return filtered_coords

    def plot_harris_points(image, filtered_coords):
        """ plots corners found in image"""
        figure()
        gray()
        imshow(image)
        plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'r*')
        axis('off')
        show()

    # Usage: 
    #harris('./path/to/image.jpg')


    # Provided code for plotting inlier matches between two images

    def plot_inlier_matches(ax, img1, img2, inliers):
        """
        Plot the matches between two images according to the matched keypoints
        :param ax: plot handle
        :param img1: left image
        :param img2: right image
        :inliers: x,y in the first image and x,y in the second image (Nx4)
        """
        res = np.hstack([img1, img2])
        ax.set_aspect('equal')
        ax.imshow(res, cmap='gray')
        
        ax.plot(inliers[:,0], inliers[:,1], '+r')
        ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
        ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
                [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
        ax.axis('off')
        
    # Usage:
    # fig, ax = plt.subplots(figsize=(20,10))
    # plot_inlier_matches(ax, img1, img2, computed_inliers)


    #######################################
    ### Your implementation starts here ###
    #######################################

    # See assignment page for the instructions!

    SIFT = True
    left = Image.open(img+"1.jpg").convert("L").convert("F")
    right = Image.open(img+"2.jpg").convert("L").convert("F")
    hl = get_harris_points(compute_harris_response(np.array(left)))       #588 points
    hr = get_harris_points(compute_harris_response(np.array(right)))
    neighbour = 8
    hl = np.array(hl)
    hr = np.array(hr)
    ld = []
    rd = []
    for ly, lx in hl:
        region = left.crop((lx-neighbour, ly-neighbour, lx+neighbour, ly+neighbour))
        descriptor = np.array(region).flatten()
        ld.append(descriptor)
    for ry, rx in hr:
        region = right.crop((rx-neighbour, ry-neighbour, rx+neighbour, ry+neighbour))
        descriptor = np.array(region).flatten()
        rd.append(descriptor)
    left = np.array(left)
    right = np.array(right)
    ld = np.array(ld)                       
    rd = np.array(rd)
    threshold = 10000
    if(SIFT):
        left = cv2.cvtColor(cv2.imread(img+"1.jpg"), cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(cv2.imread(img+"2.jpg"), cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        hl, ld = sift.detectAndCompute(left, None)
        sift = cv2.xfeatures2d.SIFT_create()
        hr, rd = sift.detectAndCompute(right, None)
        ld = np.array(ld)                       
        rd = np.array(rd)
        hl = np.array(hl)
        hr = np.array(hr)
        
    lmatch = []
    rmatch = []
    distances = scipy.spatial.distance.cdist(ld, rd, 'sqeuclidean')
    for r in range(rd.shape[0]):
        for l in range(ld.shape[0]):
            if(abs(distances[l][r]) < threshold):
            #   print("r: "+str(r)+" l: "+str(l)+" distance: "+str(distances))
                if(SIFT):
                    lmatch.append((hl[int(l)].pt[0], hl[int(l)].pt[1]))  
                    rmatch.append((hr[int(r)].pt[0], hr[int(r)].pt[1]))        
                else:
                    lmatch.append(hl[int(l)][::-1])
                    rmatch.append(hr[int(r)][::-1])
    lmatch = np.vstack(lmatch)
    rmatch = np.vstack(rmatch)
    computed_inliers = np.concatenate((lmatch, rmatch), axis=1)  

    # fig, ax = plt.subplots(figsize=(20,10))
    # plot_inlier_matches(ax, left, right, computed_inliers)
    # plt.show()

    iternum = 10000
    inlier_threshold = 0.1
    best_inliers = []
    best_H = None
    best_residual = -1
    for i in range(iternum):
        A = []
        pick = computed_inliers[sample(range(computed_inliers.shape[0]), 4)]
        for j in range(4):
            x1, y1, x2, y2 = pick[j]
            A.append([0, 0, 0, x1, y1, 1, -1*y2*x1, -1*y2*y1, -1*y2])  
            A.append([x1, y1, 1, 0, 0, 0, -1*x2*x1, -1*x2*y1, -1*x2])   
        A = np.array(A)
        U, s, V = np.linalg.svd(A)
        H = V[len(V)-1].reshape(3, 3)
        H = H / H[2, 2]   
        if np.linalg.matrix_rank(H) < 3:
            continue 
        target = np.dot(H, np.column_stack((lmatch, np.ones(len(lmatch)))).T).T
        target /= target[:, 2].reshape(-1, 1)
    #     residuals = np.sum((rmatch - target[:, :2]) ** 2, axis=1)
        residuals = np.linalg.norm(rmatch - target[:, :2], axis=1) ** 2
        inliers = np.where(residuals < inlier_threshold)[0]
        avg_residual = np.mean(residuals[inliers])   
        if len(inliers) > len(best_inliers):
            show_match = computed_inliers[inliers].copy()
            best_inliers = inliers.copy()
            best_H = H.copy()
            best_residual = avg_residual.copy()
            print(img+" number of inliers: " + str(len(best_inliers)) + "\n average residual: " + str(best_residual))


    # fig, ax = plt.subplots(figsize=(20,10))
    # plot_inlier_matches(ax, left, right, show_match)
    # plt.show()

    transform = skimage.transform.ProjectiveTransform(best_H)
    row, column = left.shape[:2]
    corners = np.array([[0, 0],[0, row],[column, 0],[column, row]])
    corners = np.vstack((transform(corners), corners))
    big = np.ceil((np.max(corners, axis=0) - np.min(corners, axis=0))[::-1])
    s = skimage.transform.SimilarityTransform(translation=-1*np.min(corners, axis=0))
    l = skimage.transform.warp(left, s.inverse, output_shape=big, cval=-1)
    r = skimage.transform.warp(left, (transform + s).inverse, output_shape=big, cval=-1)
    l0 = skimage.transform.warp(left, s.inverse, output_shape=big, cval=0)
    r0 = skimage.transform.warp(left, (transform + s).inverse, output_shape=big, cval=0)
    mask = (l != -1.0 ).astype(int) + (r != -1.0).astype(int)
    mask += (mask < 1).astype(int)
    res = np.asarray(Image.fromarray((255*(l0+r0)/mask).astype('uint8'), mode='L'))
    # cv2.imwrite(img+"res.jpg", res)