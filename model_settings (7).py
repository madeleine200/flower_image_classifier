#define function for classifier settings
def classifier_settings(classifier_input,hidden_layer):
    from torch import nn, optim
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_input, hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier


# Define a function for the validation pass
def validation(model, validloader, criterion,device):
    import torch
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
        return test_loss, accuracy

#define function for training the model
def train_model(model,epochs,print_every,trainloader, validloader,criterion,optimizer,device):
    import torch
    from torchvision import datasets, models, transforms
    from torch import nn, optim
    from torch.optim import lr_scheduler
    from torch.autograd import Variable
    
    epochs = epochs
    steps = 0
    print_every = print_every
    
    model.to(device)
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:  #only validates every n steps
            # set network to eval mode for inference
                model.eval()
            # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion,device)
                    #print running results
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                        "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
                    running_loss = 0
            
            # Turn training is back on
                model.train()
                
#define accuracy function
def check_accuracy(testloader,model,device):
    import torch
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))