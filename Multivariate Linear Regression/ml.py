import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
np.seterr(all='ignore')

#Importing the data
data = data = np.loadtxt('sample2.txt', delimiter=",")
data[:,0] = (data[:,0] - data[:,0].mean()) / (data[:,0].ptp())
data[:,1] = (data[:,1] - data[:,1].mean()) / (data[:,1].ptp())
data[:,2] = (data[:,2] - data[:,2].mean()) / (data[:,2].ptp())

#Setting up a matrix of ones
ones = np.ones((len(data),1))


#choosing random variables
theta = np.array([1000,100,100])
# This is the y vector
y = data[:,2].reshape(-1,1)
#The x matrix
x = np.concatenate((ones, data[:,0].reshape(-1,1), data[:,1].reshape(-1,1)), axis=1)

#Setting the parameters
iteration = 0
alpha = 0.05
m = 1 / (len(x))
while iteration < 10000:
    iteration += 1
    error = y.T - (theta @ x.T)
    J = m * (error @ error.T)
    dj = (m*-2) * ((error) @ x)
    theta = theta - alpha * dj

print('Theta0: {:f}'.format(theta[0][0]))
print('Theta0: {:f}'.format(theta[0][1]))
print('Theta0: {:f}'.format(theta[0][2]))
