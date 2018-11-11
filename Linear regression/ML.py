# Simple imlementation of a linear regression algorithm
import numpy as np
import matplotlib.pyplot as plt
# Import the data we will work with
data = np.loadtxt('sample.txt', delimiter=",")

#Adding a column of ones
ones = np.ones((len(data),1))
x = np.concatenate((ones, data[:,0].reshape(-1,1)), axis=1)

#Creating the y vector
y = data[:,1].reshape(-1,1)

theta = np.array([100,100])
iteration = 5000
alpha = 0.01
m = 1 / (len(x))
while iteration > 0:
    iteration -= 1
    error = y.T - (theta @ x.T)
    J = m * (error @ error.T)
    print(J)
    dj = (m*-2) * ((error) @ x)
    theta = theta - alpha * dj
print('Theta0: {:f}'.format(theta[0][0]))
print('Theta1: {:f}'.format(theta[0][1]))
