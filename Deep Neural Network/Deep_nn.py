import numpy as np
from scipy.special import expit
np.set_printoptions(suppress=True, precision=10)

#Neural Network With 6 layers
class nn_6layers(object):
    def __init__(self, N_in, N_h1, N_h2, N_h3, N_h4, N_out):
        #Numbers of nodes in each layer
        self.N_in = N_in
        self.N_h1 = N_h1
        self.N_h2 = N_h2
        self.N_h3 = N_h3
        self.N_h4 = N_h4
        self.N_out = N_out
        #Setting up the weights at random
        self.W1 = np.random.randn(self.N_h1, self.N_in) * 0.01
        self.b1 = np.zeros((self.N_h1, 1))
        self.W2 = np.random.randn(self.N_h2, self.N_h1) * 0.01
        self.b2 = np.zeros((self.N_h2, 1))
        self.W3 = np.random.randn(self.N_h3, self.N_h2) * 0.01
        self.b3 = np.zeros((self.N_h3, 1))
        self.W4 = np.random.randn(self.N_h4, self.N_h3) * 0.01
        self.b4 = np.zeros((self.N_h4, 1))
        self.W5 = np.random.randn(self.N_out, self.N_h4) * 0.01
        self.b5 = np.zeros((self.N_out, 1))

    def Train(self,X,Y,Learning_rate):
        #Forward propagation
        #First layer
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)
        #Second Layer
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = np.tanh(Z2)
        #Third layer
        Z3 = np.dot(self.W3, A2) + self.b3
        A3 = np.tanh(Z3)
        #Fourth Layer
        Z4 = np.dot(self.W4, A3) + self.b4
        A4 = np.tanh(Z4)
        #Fifth Layer
        Z5 = np.dot(self.W5, A4) + self.b5
        A5 = expit(Z5)

        #Calculate the cost
        m = len(Y)
        logprobs = (Y * np.log(A5)) + ((1- Y)*np.log(1 - A5))
        cost = (-1/m) * np.sum(logprobs)

        #Backpropagation
        M = len(X)
        #Fifth Layer
        dA5 = (- Y / A5) + ( (1 - Y)/ (1 - A5))
        dZ5 =  dA5 * (1 - A5)
        dW5 = (1/M)* (dZ5 @ A4.T) # W5
        db5 = (1/M) * np.sum(dZ5, axis = 1, keepdims = True) # b5
        #Fourth layer
        dA4 = self.W5.T @ dZ5
        dZ4 = dA4 * (1 - np.power(A4, 2))
        dW4 = (1/M)* (dZ4 @ A3.T) # W4
        db4 = (1/M) * np.sum(dZ4, axis = 1, keepdims = True) # b4
        #Third layer
        dA3 = self.W4.T @ dZ4
        dZ3 = dA3 * (1 - np.power(A3, 2))
        dW3 = (1/M)*(dZ3 @ A2.T) # W3
        db3 = (1/M) * np.sum(dZ3, axis = 1, keepdims = True) # b3
        #Second layer
        dA2 = self.W3.T @ dZ3
        dZ2 = dA2 * (1 - np.power(A2, 2))
        dW2 = (1/M)*(dZ2 @ A1.T) # W2
        db2 = (1/M) * np.sum(dZ2, axis = 1, keepdims = True) # b2
        #First layer
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * (1 - np.power(A1, 2))
        dW1 = (1/M)*(dZ1 @ X.T) #W1
        db1 = (1/M)*np.sum(dZ1, axis = 1, keepdims= True) #b1

        #Update the parameters
        #Fifth Layer
        self.W5 = self.W5 - (Learning_rate * dW5)
        self.b5 = self.b5 - (Learning_rate * db5)
        #Fourth Layer
        self.W4 = self.W4 - (Learning_rate * dW4)
        self.b4 = self.b4 - (Learning_rate * db4)
        #Third Layer
        self.W3 = self.W3 - (Learning_rate * dW3)
        self.b3 = self.b3 - (Learning_rate * db3)
        #Second Layer
        self.W2 = self.W2 - (Learning_rate * dW2)
        self.b2 = self.b2 - (Learning_rate * db2)
        #First Layer
        self.W1 = self.W1 - (Learning_rate * dW1)
        self.b1 = self.b1 - (Learning_rate * db1)


#Small test for the dimensions
model = nn_6layers(5,4,3,5,6,2)
X = np.array([[1,4,5,6,8], [1,2,3,4,5]])

Y2 = np.ones((1,2))
Y3 = np.zeros((1,2))
Y = np.vstack((Y2,Y3))

for i in range(10):
    model.Train(X.T,Y,0.01)
