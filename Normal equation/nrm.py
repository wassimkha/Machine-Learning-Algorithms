import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
np.seterr(all='ignore')
#Loading the data
data = data = np.loadtxt('sample2.txt', delimiter=",")
data[:,0] = (data[:,0] - data[:,0].mean()) / (data[:,0].ptp())
data[:,1] = (data[:,1] - data[:,1].mean()) / (data[:,1].ptp())
data[:,2] = (data[:,2] - data[:,2].mean()) / (data[:,2].ptp())
#Setting up a matrix of one
ones = np.ones((len(data),1))
#The y vector
y = data[:,2].reshape(-1,1)
#The x matrix
x = np.concatenate((ones, data[:,0].reshape(-1,1), data[:,1].reshape(-1,1)), axis=1)
#The final theta
theta = np.linalg.inv(x.T @ x) @ (x.T @ y)

print('Theta0: {:f}'.format(theta[0][0]))
print('Theta0: {:f}'.format(theta[1][0]))
print('Theta0: {:f}'.format(theta[2][0]))
