import numpy as np


#Importing the data
data = np.loadtxt('sample.txt', delimiter=",")
class LinearRegression(object):
    #Setting some of our initial parameters
    def __init__(self, alpha, theta):
        self.alpha = alpha
        self.theta = theta
    #Cleaning the data to make it ready for the training
    def Cleaningdata(self, data):
        ones = np.ones((len(data),1))
        #This needs to be modified before calibrated the method
        x = np.concatenate((ones, data[:,0].reshape(-1,1)), axis=1)
        y = data[:,1].reshape(-1,1)
        return (x, y)
    #Training the model using gradient descent
    def train(self,x , y, m):
        error = y.T - (self.theta @ x.T)
        J = m * (error @ error.T)
        dj = (m*-2) * ((error) @ x)
        self.theta = self.theta - self.alpha * dj
    #We can also train our model using the normal equation
    def normalEquation(self, x, y):
        self.theta = np.linalg.inv(x.T @ x) @ (x.T @ y)
    #We can query a specific answer
    def query(self, x):
        return self.theta @ x

#Example of a model
model = LinearRegression(0.00000001, [10,10])
x, y = model.Cleaningdata(data)

#Setting our training loop
iteration = 0
m = 1/len(data)
while iteration < 100000:
    iteration += 1
    model.train(x,y,m)
#Asking the model to answer a specific question
answer = model.query([100,23])

#Example of model traning using the normal equation
model.normalEquation(x,y)

