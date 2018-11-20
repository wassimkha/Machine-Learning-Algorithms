from scipy.special import expit
import numpy as np


class NeuralNetwork(object):
    #This is a general neural network with 3 layers
    def __init__(self,n_inputs,n_hidden, n_out):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_out = n_out
        #Setting up the weights at random between -0.5 and 0.5
        self.wih = np.random.rand(self.n_hidden, self.n_inputs) - 0.5
        self.who = np.random.rand(self.n_out, self.n_hidden) - 0.5

    def train(self, input, y, alpha):
        input = np.array(input).reshape((len(input),1))
        y = np.array(y).reshape((len(y),1))
        #Calculate the Hidden's matrix output
        input_hidden = self.wih @ input
        hidden_activation = expit(input_hidden)
        #Calculate the Output's response
        hidden_output = self.who @ hidden_activation
        output_activation = expit(hidden_output)
        #Calculate the Errors
        error_output = ( y - output_activation)
        error_hidden =  self.who.T @ error_output
        #Recalibrating the weights
        self.who += alpha * ((error_output * output_activation * (1 - output_activation)) @ hidden_activation.T)
        self.wih += alpha * ((error_hidden * hidden_activation * (1 - hidden_activation) @ input.T))
    def query(self, input):
        input = np.array(input).reshape((len(input),1))
        #Calculate the Hidden's matrix output
        input_hidden = self.wih @ input
        hidden_activation = expit(input_hidden)
        #Calculate the Output's response
        hidden_output = self.who @ hidden_activation
        output_activation = expit(hidden_output)
        return output_activation

#Importing and cleaning and seperating the data
data = np.loadtxt(open("mnist_train.csv", "rb"), delimiter=",")
target = []
image = []
for i in data:
    target.append(i[0])
    image.append(i[1:])
for k in range(len(image)):
    image[k] = image[k]/256
for j in range(len(target)):
    number = target[j]
    target[j] = np.zeros(10) + 0.01
    target[j][int(number)] = 0.99

#Making our neural network model
model = NeuralNetwork(784,500,10)

#Training our NeuralNetwork
#Going trought the training data 5 times
epoch = 0
while epoch < 5:
    epoch += 1
    for l in range(len(target)):
        model.train(image[l], target[l], 0.3)

#Testing our network
test = np.loadtxt(open("mnist_test.csv", "rb"), delimiter=",")
target_test = []
image_test = []
for o in test:
    target_test.append(o[0])
    image_test.append(o[1:])
for t in range(len(test)):
    image_test[t] = image_test[t]/256
for h in range(len(test)):
    number = target_test[h]
    target_test[h] = np.zeros(10) + 0.01
    target_test[h][int(number)] = 0.99
result = 0
for s in range(len(target_test)):
    output = model.query(image_test[s])
    compare = target_test[s]
    if np.argmax(compare) == np.argmax(output):
        result += 1
result /= 100
print('Accuracy: {:f} %'.format(result))
