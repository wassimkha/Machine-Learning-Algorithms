import torch
import torch.nn as nn
import torchvision.transforms as transforms
import collections
import torchvision.datasets as datasets
#Importing data
Train_dataset = datasets.MNIST(root='./data',train = True, transform = transforms.ToTensor(),download=False)
Test_dataset = datasets.MNIST(root='./data',train = False, transform = transforms.ToTensor())
#Cleaning the data and separating it
Batch_size = 100
n_iters = 10000
N_epochs = int(n_iters /( len(Train_dataset) / Batch_size))
#Making sure it is iterable
Train_loader = torch.utils.data.DataLoader(dataset=Train_dataset,batch_size=Batch_size,shuffle=True)
print(isinstance(Train_loader, collections.Iterable))
#Cleaning the training data
Test_loader = torch.utils.data.DataLoader(dataset=Test_dataset,batch_size=Batch_size,shuffle=True)
print(isinstance(Test_loader, collections.Iterable))


#Build the model, The same as linear regression
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

#Choosing the dimension of our model
input_dim = 784
output_dim = 10
model = LogisticRegressionModel(input_dim,output_dim)
#Loss class
criterion = nn.CrossEntropyLoss()

#Optimizer class
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
iter = 0
#Trainig the model
for i in range(N_epochs):
    for j, (images,labels) in enumerate(Train_loader):
        iter += 1
        images = images.view(-1,28*28).requires_grad_()
        labels = labels

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        #Calculating the accuracy
        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in Test_loader:
                images = images.view(-1,28*28).requires_grad_()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            accuracy = float(100 * (correct.item() / total))

            print('Loss: {}. Accuracy: {}'.format(loss.data.item(), accuracy))


#Saving and loading the model
saveModel = False
if saveModel:
    torch.save(model.state_dict(), 'awesome.pkl')

loadModel = False
if loadModel:
    model.load_state_dict(torch.load('awesome.pkl'))
