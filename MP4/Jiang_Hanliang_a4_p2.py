# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
## Fundamental Matrix Estimation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt 
from random import sample
import mp3code

##
## load images and match files for the first example
##

def fundamental(image_genre = "library", normalize = False):

    if image_genre == "library":
        I1 = Image.open('MP3_part2_data/library1.jpg')
        I2 = Image.open('MP3_part2_data/library2.jpg')
        matches = np.loadtxt('MP3_part2_data/library_matches.txt')
    if image_genre == "lab":
        I1 = Image.open('MP3_part2_data/lab1.jpg')
        I2 = Image.open('MP3_part2_data/lab2.jpg')
        matches = np.loadtxt('MP3_part2_data/lab_matches.txt')


    # this is a N x 4 file where the first two numbers of each row
    # are coordinates of corners in the first image and the last two
    # are coordinates of corresponding corners in the second image: 
    # matches(i,1:2) is a point in the first image
    # matches(i,3:4) is a corresponding point in the second image

    N = len(matches)

    ##
    ## display two images side-by-side with matches
    ## this code is to help you visualize the matches, you don't need
    ## to use it to produce the results for the assignment
    ##

    I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
    I3[:,:I1.size[0],:] = I1
    I3[:,I1.size[0]:,:] = I2
    I3 = I3/255.0
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # ax.imshow(np.array(I3).astype(float))
    # ax.plot(matches[:,0],matches[:,1],  '+r')
    # ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
    # ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
    # plt.show()

    ##
    ## display second image with epipolar lines reprojected 
    ## from the first image
    ##

    def norm(cord):
        mean = np.mean(cord, axis=0)
        xstd = np.std(cord[:,0])
        ystd = np.std(cord[:,1])
        Tmatrix = np.array([[sqrt(2)/xstd, 0, -sqrt(2)/xstd*mean[0]], [0, sqrt(2)/ystd, -sqrt(2)/ystd*mean[1]], [0, 0, 1]])
        cord = np.concatenate((cord, np.ones((cord.shape[0],1))), axis=1)
        # cord.shape = N*3
        cord = np.dot(Tmatrix, cord.T).T
        return Tmatrix, cord[:,0:2]

    def fit_fundamental(matches, normalize=True):
        left = matches[:,0:2]
        right = matches[:,2:4]
        if normalize:
            T1, left = norm(left)
            T2, right = norm(right)
        eight_points = sample(range(left.shape[0]), 8)
        left_selected = left[eight_points]
        right_selected = right[eight_points]
        Umatrix = []
        for i in range(8):
            x1 = left_selected[i][0]
            y1 = left_selected[i][1]
            x2 = right_selected[i][0]
            y2 = right_selected[i][1]
            # 8 U vector 
            Umatrix.append([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
        Umatrix = np.array(Umatrix)
        U, S, V = np.linalg.svd(Umatrix)
        F = V[-1].reshape(3, 3)
        F = F/F[2][2]
        U, S, V = np.linalg.svd(F)
        #rank - 2 constraint
        diag = np.diag(S)
        diag[-1]=0
        if not normalize:
            F = np.dot(U, np.dot(diag, V))
        else:
            F = np.dot(np.dot(T2.T, np.dot(U, np.dot(diag, V))), T1)
        return F
            
    def residual(matches, F):
        left = matches[:,0:2]
        right = matches[:,2:4]
        result = 0
        N = left.shape[0]
        left = np.concatenate((left, np.ones((N,1))), axis=1)
        right = np.concatenate((right, np.ones((N,1))), axis=1)
        for i in range(N):
            result += abs(np.dot(np.dot(right[i], F), left[i].T))
        return result/N
        

    # first, fit fundamental matrix to the matches
    F = fit_fundamental(matches, normalize=normalize); # this is a function that you should write
    # print(f"residual for normalize = {normalize} algorithm of {image_genre} is {residual(matches, F)}")
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
    closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

    # display points and segments of corresponding epipolar lines
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # ax.imshow(np.array(I2).astype(float)/255.0)
    # ax.plot(matches[:,2],matches[:,3],  '+r')
    # ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
    # ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')
    # plt.show()


# Camera Calibration
def lab_matrix():
    def evaluate_points(M, points_2d, points_3d):
        """
        Visualize the actual 2D points and the projected 2D points calculated from
        the projection matrix
        You do not need to modify anything in this function, although you can if you
        want to
        :param M: projection matrix 3 x 4
        :param points_2d: 2D points N x 2
        :param points_3d: 3D points N x 3
        :return:
        """
        N = len(points_3d)
        points_3d = np.hstack((points_3d, np.ones((N, 1))))
        points_3d_proj = np.dot(M, points_3d.T).T
        u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
        v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
        residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
        points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
        return points_3d_proj, residual

    def cam_calib(points, matches):
        N = matches.shape[0]
        points = np.concatenate((points, np.ones((N,1))), axis=1)
        Umatrix = []
        for i in range(N):
            Umatrix.append(np.array([0, 0, 0, 0, points[i][0], points[i][1], points[i][2], points[i][3], -matches[i][1]*points[i][0], -matches[i][1]*points[i][1], -matches[i][1]*points[i][2], -matches[i][1]*points[i][3]]))
            Umatrix.append(np.array([points[i][0], points[i][1], points[i][2], points[i][3], 0, 0, 0, 0, -matches[i][0]*points[i][0], -matches[i][0]*points[i][1], -matches[i][0]*points[i][2], -matches[i][0]*points[i][3]]))
        Umatrix = np.array(Umatrix)
        U, S, V = np.linalg.svd(Umatrix)
        V = V[-1].reshape(3, 4)
        return V/V[2][2]
        

        
        
    points = np.loadtxt('MP3_part2_data/lab_3d.txt')
    matches = np.loadtxt('MP3_part2_data/lab_matches.txt')
    calib1= cam_calib(points, matches[:,0:2])
    calib2= cam_calib(points, matches[:,2:4])
    # print(f"left camera matrix: \n{calib1}\n")
    # print(f"right camera matrix: \n{calib2}")
    _, residual1 = evaluate_points(calib1, matches[:,0:2], points)
    _, residual2 = evaluate_points(calib2, matches[:,2:4], points)
    # print(f"left residual: {residual1}")
    # print(f"right residual: {residual2}")
    return calib1, calib2


def center_trangulate():
    ## Camera Centers
    def evaluate_points(M, points_2d, points_3d):
            """
            Visualize the actual 2D points and the projected 2D points calculated from
            the projection matrix
            You do not need to modify anything in this function, although you can if you
            want to
            :param M: projection matrix 3 x 4
            :param points_2d: 2D points N x 2
            :param points_3d: 3D points N x 3
            :return:
            """
            N = len(points_3d)
            points_3d = np.hstack((points_3d, np.ones((N, 1))))
            points_3d_proj = np.dot(M, points_3d.T).T
            u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
            v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
            residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
            points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
            return points_3d_proj, residual
        
    def center(calib):
        U, S, V = np.linalg.svd(calib)
        V = V[-1]
        return V/V[-1]

    lab_points = np.loadtxt('MP3_part2_data/lab_3d.txt')
    lab_matches = np.loadtxt('MP3_part2_data/lab_matches.txt')
    library_matches = np.loadtxt('MP3_part2_data/library_matches.txt')
    lab_calib1, lab_calib2 = lab_matrix()
    library_calib1 = np.loadtxt('MP3_part2_data/library1_camera.txt')
    library_calib2 = np.loadtxt('MP3_part2_data/library2_camera.txt')
    lab_center1 = center(lab_calib1)
    lab_center2 = center(lab_calib2)
    library_center1 = center(library_calib1)
    library_center2 = center(library_calib2)
    print(f"lab left center coordinate: {lab_center1}")
    print(f"lab right center coordinate: {lab_center2}")
    print(f"library left center coordinate: {library_center1}")
    print(f"library right center coordinate: {library_center2}")
    ## Triangulation
    def trangulation(match, left, right):
        N = match.shape[0]
        result = np.zeros((N,3))
        for i in range(N):
            matrix1 = np.dot(np.array([[0, -1, match[i][1]],
                                    [1, 0, -match[i][0]],
                                    [-match[i][1], match[i][0], 0]]), left)
            matrix2 = np.dot(np.array([[0, -1, match[i][3]],
                                    [1, 0, -match[i][2]],
                                    [-match[i][3], match[i][2], 0]]), right)
            matrix = np.concatenate((matrix1, matrix2), axis=0)
            U, S, V = np.linalg.svd(matrix)
            V = V[-1]
            V = V/V[-1]
            result[i] = V[:3]
        return result

    # display lab
    lab_3dpoints = trangulation(lab_matches, lab_calib1, lab_calib2)
    _, residual1 = evaluate_points(lab_calib1, lab_matches[:,0:2], lab_3dpoints)
    _, residual2 = evaluate_points(lab_calib2, lab_matches[:,2:4], lab_3dpoints)
    print(f"lab residual for left is {np.mean(residual1)}")
    print(f"lab residual for right is {np.mean(residual2)}")
    lab_center = np.concatenate((lab_center1[np.newaxis,:], lab_center2[np.newaxis,:]), axis=0)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(lab_3dpoints[:,0], lab_3dpoints[:,1], lab_3dpoints[:,2], c='c')
    ax.scatter(lab_center[:,0], lab_center[:,1], lab_center[:,2], c='k')
    plt.show()

    # display library
    library_3dpoints = trangulation(library_matches, library_calib1, library_calib2)
    _, library_residual1 = evaluate_points(library_calib1, library_matches[:,0:2], library_3dpoints)
    _, library_residual2 = evaluate_points(library_calib2, library_matches[:,2:4], library_3dpoints)
    print(f"library residual for left is {library_residual1}")
    print(f"library residual for right is {library_residual2}")
    library_center = np.concatenate((library_center1[np.newaxis,:], library_center2[np.newaxis,:]), axis=0)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(library_3dpoints[:,0], library_3dpoints[:,1], library_3dpoints[:,2], c='c')
    ax.scatter(library_center[:,0], library_center[:,1], library_center[:,2], c='k')
    plt.show()


# fundamental(image_genre = "library", normalize = False)
# lab_matrix()
# center_trangulate
mp3code.code('MP3_part2_data/gaudi')
mp3code.code('MP3_part2_data/house')