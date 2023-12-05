import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

matrix = np.loadtxt('factorization_data/measurement_matrix.txt')
mean_matrix = np.mean(matrix, axis=1).reshape(202,1)
measurement_matrix = matrix - mean_matrix
U, sigma, V = np.linalg.svd(measurement_matrix)
U = U[:,0:3]
sigma = np.diag(sigma)[0:3,0:3]
V = V[0:3,:]
Q = np.sqrt(sigma)
M = np.dot(U, Q)
S = np.dot(Q, V)
original = np.dot(M, S)
I = np.identity(M.shape[0])
L = np.dot(np.dot(np.matrix(M).I, I), np.matrix(M.T).I)
Q = np.linalg.cholesky(L)
print(f"matrix Q: \n{Q}")
S = np.dot(Q.I, S).T
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(S[:,0], S[:,1], S[:,2], c="r")
plt.show()
frame = [30, 60, 90]
for i in range(3):
          im = image.imread('factorization_data/frame000000' + str(frame[i]) + '.jpg')
          fig, ax = plt.subplots()
          ax.imshow(im)
          ax.plot(matrix[2*frame[i]-2,:], matrix[2*frame[i]-1,:], '+m')
          ax.plot(original[2*frame[i]-2,:]+mean_matrix[2*frame[i]-2,0], original[2*frame[i]-1,:]+mean_matrix[2*frame[i]-1,0], '+y')
          plt.show()
frame_num = measurement_matrix.shape[0]//2
residual = np.zeros(frame_num)
for i in range(frame_num):
          for j in range(measurement_matrix.shape[1]):
                    residual[i] = np.linalg.norm(np.array([original[2*i][j]-measurement_matrix[2*i][j], original[2*i+1][j]-measurement_matrix[2*i+1][j]]))

print(f"residual: {np.sum(residual)}")
cord = range(frame_num)
plt.figure()
plt.plot(cord, residual)
plt.show()