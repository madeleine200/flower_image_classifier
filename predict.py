import argparse
from torchvision import datasets, models, transforms
from collections import OrderedDict
import numpy as np
import pandas as pd
import json
import torch
from torch import nn, optim
from torch.autograd import Variable
from utility import process_image, label_map
from model_settings  import classifier_settings
def main():
   
    # Set parameters via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='ImageClassifier/flowers/',
                     help='path to the folder of flower images')
    parser.add_argument('--checkpoint_path', default = 'ImageClassifier/checkpoint.pth' )
    parser.add_argument('--topk', default = 1 )
    parser.add_argument('--device',default="cpu")
    parser.add_argument('--cat_names', default='ImageClassifier/cat_to_name.json')
    args=parser.parse_args()
    
    image_path=args.image_path
    checkpoint_path=args.checkpoint_path
    topk=int(args.topk)
    device=args.device
    cat_names=args.cat_names
    
    #Load model settings
    chpt = torch.load(checkpoint_path)
    model_arch=chpt['arch']
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
    
    if model_arch=='vgg':
        model = models.vgg19(pretrained=True)
        classifier_input=25088
    elif model_arch=='alexnet':
        model=models.alexnet(pretrained=True)
        classifier_input=9216
    
    
    model.class_to_idx = chpt['class_to_idx']
    hidden_layer=chpt['hidden_layer']
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_input, hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier=classifier
    model.load_state_dict(chpt['state_dict'])
    
    #Process image 
    img=process_image(image_path)
    img_tensor=Variable(torch.FloatTensor(img),requires_grad=True)
    if device=='cuda' and torch.cuda.is_available:
        model.cuda()
        img_tensor=img_tensor.cuda()
         
    #add batch number so that model shape is [1,3,224,224]
    model_input = img_tensor.unsqueeze(0)
    
    #calculate probabilites from model
    probs=torch.exp(model.forward(model_input))
    
    #top probabilities
    top_probs,top_labels=probs.topk(topk)
    #Send tensors to cpu
    top_probs=top_probs.cpu()
    top_labels=top_labels.cpu()
    #Convert to numpy array 
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labels = top_labels.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    
    labels = [idx_to_class[label] for label in top_labels]
    
    top_names = [cat_to_name[idx_to_class[label]] for label in top_labels]
    #Print results
    for i in range(topk):
        print("Flower type: {}  Probability: {:3f}".format(top_names[i],top_probs[i]))
        
if __name__ == "__main__":
    main()
