import torch
import torch.nn as nn
# import torch.autograd as Variable
import numpy as np

#Pytorch Linear regression
x = [i for i in range(11)]
y = [2*i + 1 for i in x]
x = np.array(x, dtype=np.float32).reshape((-1,1))
y = np.array(y, dtype=np.float32).reshape((-1,1))

#Build the model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


#Model and dmensions
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
#Loss function
criterion = nn.MSELoss()

learning_rate = 0.001
#The graadient descent function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#Inputs and Outputs
I = torch.from_numpy(x).requires_grad_()
Y = torch.from_numpy(y).requires_grad_()
#Training
for i in range(1000):
    optimizer.zero_grad()
    outputs = model(I)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

#Prediction
predicted = model(I).data.numpy()
print('prediction:')
print(predicted.reshape((1,-1)))
print('target:')
print(Y.data.numpy().reshape((1,-1)))
#Saving and loading the model
saveModel = False
if saveModel:
    torch.save(model.state_dict(), 'awesome.pkl')

loadModel = False
if loadModel:
    model.load_state_dict(torch.load('awesome.pkl'))
