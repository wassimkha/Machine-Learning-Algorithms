# Simple imlementation of a linear regression algorithm
import numpy as np
import matplotlib.pyplot as plt
# Import the data we will work with
data = np.loadtxt('sample.txt', delimiter=",")

#Setting up the starting variables
theta_0 = 120
theta_1 = 120
alpha = 0.02
iteration = 0
m2 = 1 / (data.shape[0])
m = 1 / (2*data.shape[0])

#Setting up the iteration loop
while iteration < 3000:
    prediction_cost_function = 0
    prediction_theta_0 = 0
    prediction_theta_1 = 0
    # cost function
    for i in range(data.shape[0]):
        h = (data[i,0]*theta_1 + theta_0)
        prediction_theta_0 += (h-data[i,1])
        prediction_theta_1 += ((h-data[i,1])*data[i,0])
        prediction_cost_function += (h-data[i,1])**2
    J = m*prediction_cost_function # This is the cost function (or error function)
    # Gradient descent
    #theta_0
    theta_0 = theta_0 - (alpha*m2*prediction_theta_0)
    #theta_one
    theta_1 = theta_1 - (alpha*m2*prediction_theta_1)

    iteration += 1
    print('Cost function value: '+ str(J))
    print('Theta 0: '+ str(theta_0))
    print('Theta 1: '+ str(theta_1))

#Printing the values to graph
plt.plot([0,25], [theta_0, theta_1*25+theta_0])
plt.scatter(data[:,0], data[:,1])
plt.show()
