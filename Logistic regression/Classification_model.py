#Import the required libraries and the sigmoid function(expit)
from scipy.special import expit
import numpy as np
import math

#Import our data
data = np.array([[1,0],[2,0],[3,0],[4,1],[5,1],[6,1],[7,1]])

class Classification(object):
    #Setting some of our parameters
    def __init__(self, alpha, num_parameter, data):
        self.alpha = alpha
        self.num_parameter = np.zeros((num_parameter, 1))
        self.data = data
        self.m = 1 / len(data)
    #This method cleans our data and make it ready for the training
    def cleanData(self):
        ones = np.ones((len(data),1))
        x = np.concatenate((ones, data[:,0].reshape(-1,1)), axis=1)
        y = data[:,1].reshape(-1,1)
        return (x,y)
    #We can train our model using gradient descent
    def train(self,x, y):
        z = x @ self.num_parameter
        h = expit(z).reshape((-1,1))
        j = self.m * ((-y.T @ np.log(h)) - (((1 - y).T) @ (np.log(1-h))))
        dj = (self.alpha/self.m) * (x.T @ (h - y))
        self.num_parameter = self.num_parameter - dj
    #We can query our model for a specific answer
    def query(self, x):
        answer = x @ self.num_parameter
        answer = expit(answer)
        return(answer)

#Exaple use of a model
model = Classification(0.01, 2, data)
#Cleaning the data
x,y = model.cleanData()
#Training loop
iteration = 0
while iteration < 1000:
    iteration += 1
    model.train(x,y)

#Asking an answer for a specific problem
answer = model.query([-0,-0])
