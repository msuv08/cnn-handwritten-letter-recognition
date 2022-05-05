# Run these two blocks to load important libraries and set things up
import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import torch.nn.functional as F

import torchvision.transforms as transforms

'''
Functions dealing with data initialization and retrieval
'''

# creates a data loader given an array of data and a matching array of labels
def create_loader(data, label, batch_size=64):
    # unsqueeze first dimension since data shape: (x, 28, 28) convert to torch of shape (x, 1, 28, 28)
    data_set = torch.utils.data.TensorDataset(torch.Tensor(data).unsqueeze(dim=1), torch.Tensor(label))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader

# get a random input from a data loader
def get_input(data_loader):
    for inputs, labels in data_loader:
        img = inputs[0].unsqueeze(0)
        label = int(labels[0].item())
        return img, label

# combines two data loader into a single one
# fromEMNIST specified whether or not loader_1 is from EMNIST and needs to be handled differently
def combine_loaders(loader_1, loader_2, batch_size=64, fromEMNIST=False):
    combined_data = []
    combined_label = []
    
    for inputs, labels in loader_1:
        for i in range(inputs.shape[0]):
            if (fromEMNIST):
                combined_data += inputs[i].detach().transpose(1, 2).tolist()
            else:
                combined_data += inputs[i].detach().tolist()
            combined_label.append(labels[i].item())

    for inputs, labels in loader_2:
        for i in range(inputs.shape[0]):
            combined_data += inputs[i].detach().tolist()
            combined_label.append(labels[i].item())
    
    return create_loader(combined_data, combined_label, batch_size=batch_size)

'''
Functions to train and test our different models
'''
def train_network(model, train_loader, val_loader, criterion, optimizer, nepoch=100, display_losses=True, early_stopping=False):
    prev_loss = 69
    trigger = 0
    patience = 2
    try:
        for epoch in tqdm(range(nepoch)):
            total_loss_train = 0
            count_train = 0
            # training
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                labels = labels.type(torch.long)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss_train += loss.item()
                count_train += 1
            # validation
            with torch.no_grad():
                total_loss_val = 0
                count_val = 0
                for inputs, labels in val_loader:
                    outputs = model.forward(inputs)
                    labels = labels.type(torch.long)
                    loss = criterion(outputs, labels)
                    total_loss_val += loss.item()
                    count_val += 1

            # print epochs if indicated
            if display_losses:
                print('EPOCH %d'%epoch)
                print('{:>12s} {:>7.5f}'.format('Train loss:', total_loss_train/count_train))
                print('{:>12s} {:>7.5f}'.format('Val loss:', total_loss_val/count_val))
                print()   

            # check early stopping criterion     
            if early_stopping:
                if total_loss_val - prev_loss > -0.01:
                    trigger += 1
                    if trigger >= patience:
                        # implementing early stopping for regularization
                        print('Early stopping in epoch ' + str(epoch) + '.')
                        break
                else:
                    trigger = 0
                prev_loss = total_loss_val 


    except KeyboardInterrupt:
        print('Exiting from training early')
    return

def test_network(model, test_loader):
    correct = 0
    total = 0
    true, pred = [], []
    with torch.no_grad():
        for inputs, labels  in test_loader:
            outputs = model.forward(inputs)
            predicted = torch.argmax(outputs, dim=1) # get predicted class label for each test example.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true.append(labels)
            pred.append(predicted)
    acc = (100 * correct / total)
    true = np.concatenate(true)
    pred = np.concatenate(pred)
    return acc, true, pred

def display_contingency_matrix(num_classes, pred, true):
    contingency = np.zeros((num_classes, num_classes))
    for i in range(len(pred)):
        contingency[int(pred[i])][int(true[i])] += 1

    plt.matshow(contingency)
    plt.xlabel("true")
    plt.ylabel("predictions")
    plt.colorbar()



'''
Functions to analyze the neural network
'''
from math import ceil

# class to train optimize inputs
class OptimizeInput(nn.Module):
    def __init__(self, learned_model, input_shape=(1, 1, 28, 28)):
        super().__init__()
        self.learned_model = learned_model
        self.optimized_input = torch.zeros(input_shape, requires_grad=True) # intialization of an all 0 image
        
    def forward(self):
        return self.learned_model(self.optimized_input)
    
    def parameters(self):
        return [self.optimized_input]

# given a model, returns a dictionary mapping each letter to its optimized input
def get_optimized_inputs(model, nepochs = 10000, validate_loss=False):
    targets = torch.arange(len(model.classes)).long()
    optimized_inputs = {}

    for t in targets:
        # train input
        input_model = OptimizeInput(model)
        optimizer = torch.optim.Adam(input_model.parameters(), lr=.01)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(nepochs):
            optimizer.zero_grad()
            outputs = input_model()
            loss = criterion(outputs, t.reshape(1))
            loss.backward()
            optimizer.step()
        optimized_inputs[model.classes[t]] = input_model.optimized_input.detach().numpy().reshape((28, 28))
        
        # test input
        if (validate_loss):
            print("finished %s with %0.5f loss" % (model.classes[t], loss.item()), end = "\t")
            tensor = torch.Tensor(optimized_inputs[model.classes[t]].reshape(1, 1, 28, 28))
            with torch.no_grad():
                output = model(tensor)
            pred = model.classes[torch.argmax(output, dim=1)]
            print(f"Expected: {model.classes[t]}. Predicted: {pred}.")
    return optimized_inputs

# for each letter in the alphabet, plots the optimized input
def display_optimized_inputs(optimized_inputs):
    num_inputs = len(optimized_inputs)
    # "unnecessary code" to get square-like dimensions to align different kernels
    cols = ceil(num_inputs / int(num_inputs ** 0.5))
    rows = ceil(num_inputs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(9,9))
    count = 0
    # plots the input for each letter
    for letter in optimized_inputs:
        r = count // cols
        c = count % cols
        img = optimized_inputs[letter]
        axes[r,c].matshow(img, cmap='gray')
        title = 'opt ' + letter
        count += 1
        axes[r,c].set_title(title)
    # deletes the unnecessary cells at the end
    for i in range(count, rows*cols):
        r = i // cols
        c = i % cols
        fig.delaxes(axes[r, c])
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()   

# display the different kernels given a numpy array of weights
def display_kernels(conv_weights):
    num_kernels = conv_weights.shape[0]
    # "unnecessary code" to get square-like dimensions to align different kernels
    cols = ceil(num_kernels / int(num_kernels ** 0.5))
    rows = ceil(num_kernels / cols)
    
    # display each kernel
    fig, axes = plt.subplots(rows, cols, figsize=(9,9))
    kernel = 0
    for i in range(rows):
        for j in range(cols):
            out = conv_weights[kernel][0]
            axes[i,j].matshow(out, cmap='gray')
            kernel += 1
            title = 'Kernel ' + str(kernel)
            axes[i,j].set_title(title)
            if kernel >= num_kernels: break
    # deletes the unnecessary cells at the end
    for i in range(kernel, rows*cols):
        r = i // cols
        c = i % cols
        fig.delaxes(axes[r, c])
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()

# convolves the image and displays this convolution per kernel
def display_conv_outputs(conv_layer, img):
    num_kernels = conv_layer.weight.shape[0]
    outputs = conv_layer(img)
    # "unnecessary code" to get square-like dimensions to align different kernels
    cols = num_kernels // int(num_kernels ** 0.5)
    rows = ceil(num_kernels / cols)
    
    # display convolved input
    fig, axes = plt.subplots(rows, cols, figsize=(9,9))
    kernel = 0
    for i in range(rows):
        for j in range(cols):
            out = outputs[0][kernel]
            axes[i,j].matshow(out.detach().numpy(), cmap='gray')
            kernel += 1
            title = 'Conv ' + str(kernel)
            axes[i,j].set_title(title)
            if kernel >= num_kernels: break
    # deletes the unnecessary cells at the end
    for i in range(kernel, rows*cols):
        r = i // cols
        c = i % cols
        fig.delaxes(axes[r, c])
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()
    return outputs

# Given a pooling layer and outputs from a 2d convolution layer, shows how the input is transformed
def display_pool_outputs(pool_layer, conv_output):
    num_kernels = conv_output.shape[1]
    outputs = pool_layer(conv_output)
    # "unnecessary code" to get square-like dimensions to align different kernels
    cols = num_kernels // int(num_kernels ** 0.5)
    rows = ceil(num_kernels / cols)
    
    # display convolved input
    fig, axes = plt.subplots(rows, cols, figsize=(9,9))
    kernel = 0
    for i in range(rows):
        for j in range(cols):
            out = outputs[0][kernel]
            axes[i,j].matshow(out.detach().numpy(), cmap='gray')
            kernel += 1
            title = 'Pool ' + str(kernel)
            axes[i,j].set_title(title)
            if kernel >= num_kernels: break
    # deletes the unnecessary cells at the end
    for i in range(kernel, rows*cols):
        r = i // cols
        c = i % cols
        fig.delaxes(axes[r, c])
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()
    return outputs