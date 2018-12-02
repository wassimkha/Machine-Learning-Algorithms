import torch
import torch.nn as nn
import torchvision.transforms as transforms
import collections
import torchvision.datasets as datasets
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

#Build the model, Four Layer Neural network
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(NeuralNetworkModel, self).__init__()

        #Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu = nn.ReLU() #Activation

        #Second layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU() #Activation

        #Third layer
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU() #Activation

        #Last layer
        self.fc5 = nn.Linear(hidden_dim3,output_dim)

    def forward(self, x):
        #First layer
        out = self.fc1(x)
        out = self.relu(out) #Activation

        #Second layer
        out = self.fc2(out)
        out = self.relu2(out) #Activation

        #Third layer
        out = self.fc3(out)
        out = self.relu3(out) #Activation

        #Last layer
        out = self.fc5(out)
        return out

#Choosing the dimension of our model
input_dim = 784
hidden_dim1 = 500
hidden_dim2 = 300
hidden_dim3 = 100
output_dim = 10
model = NeuralNetworkModel(input_dim,hidden_dim1,hidden_dim2,hidden_dim3,output_dim)
#Loss class
criterion = nn.CrossEntropyLoss()
#Optimizer class
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
iter = 0
#Training the Model
for i in range(N_epochs):
    for j, (images,labels) in enumerate(Train_loader):
        iter += 1
        #Loads the images
        images = images.view(-1,28*28).requires_grad_()
        labels = labels
        #Clear the gradient from the previous iteration
        optimizer.zero_grad()
        #Feed Forward
        outputs = model(images)
        #Calculate the Loss
        loss = criterion(outputs,labels)
        #Getting the derivation of the gradients
        loss.backward()
        #Update the parameters
        optimizer.step()

        #Calculating the accuracy
        if iter % 500 == 0:
            correct = 0
            total = 0
            #Iterating trought the test Set
            for images, labels in Test_loader:
                #Load the images
                images = images.view(-1,28*28).requires_grad_()
                #Forward pass
                outputs = model(images)
                #Get the prediction from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                #Calculate the accuracy
                total += labels.size(0)
                correct += (predicted == labels).sum()

            accuracy = float(100 * (correct.item() / total))

            print('Loss: {}. Accuracy: {}'.format(loss.data.item(), accuracy))
