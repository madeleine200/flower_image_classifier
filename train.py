import argparse
import time
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
#Utility files
from utility import import_files, data_load, label_map
from model_settings import classifier_settings, validation, train_model, check_accuracy
def main():
   
     # Set parameters via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='ImageClassifier/flowers',
                     help='path to the folder of flower images')
    parser.add_argument('--arch', default = 'vgg' )
    parser.add_argument('--learning_rate', default = 0.001 )
    parser.add_argument('--hidden_layer_size', default = 4096 )
    parser.add_argument('--epochs', default = 3)
    parser.add_argument('--device', default = 'cuda')
    
    args=parser.parse_args()
    
    data_dir=args.data_dir
    learning_rate=float(args.learning_rate)
    model_arch=args.arch
    epochs=int(args.epochs)
    hidden_layer=args.hidden_layer_size
    device=args.device
    
#Import files and lable map
    train_data, valid_data,test_data=import_files(data_dir)
    trainloader,validloader,testloader=data_load(train_data,valid_data,test_data)
    label_map()

    if model_arch=='vgg':
        model = models.vgg19(pretrained=True)
        classifier_input=25088
    elif model_arch=='alexnet':
        model=models.alexnet(pretrained=True)
        classifier_input=9216
    
    # Freeze parameters in model
    for param in model.parameters():
        param.requires_grad = False
    #Set model parameters
    classifier=classifier_settings(classifier_input,hidden_layer)
    model.classifier=classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    #train model and check accuracy 
    train_model(model,epochs,40,trainloader, validloader, criterion,optimizer,device)
    check_accuracy(trainloader,model,device)
#Save model 
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    torch.save({'arch': model_arch,
                'hidden_layer':hidden_layer,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            'ImageClassifier/checkpoint.pth')

if __name__ == "__main__":
    main()
