#Import and Transform Data 
def import_files(data_dir):
    from torchvision import datasets, models, transforms
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define transforms for datasets
    train_transforms =transforms.Compose([transforms.RandomRotation(30),  transforms.RandomResizedCrop(224),  transforms.RandomHorizontalFlip(),  transforms.ToTensor(), transforms.Normalize([0.485,     0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) 
    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])                                    
     #Import data and apply transforms                                
    train_data = datasets.ImageFolder(data_dir + '/train',transform=train_transforms) 
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)                            
           
    return train_data, valid_data, test_data
#Function for loading data
def data_load(train_data,valid_data,test_data):
    import torch
    from torch import nn, optim
    from torch.optim import lr_scheduler
    from torch.autograd import Variable 

#Define data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return trainloader, validloader,testloader
#Function for Label Mapping
def label_map():
    import json
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

#function for image processing
def process_image(image_path):
    from PIL import Image
    import numpy as np
    img = Image.open(image_path)
    width,height=img.size
    
    #Resize image based on smallest side
    if width > height:
        img=img.resize((int((256/height)*width),256))
    else:
        img=img.resize((256,int((256/width)*height)))

    #Set dimensions to crop
    left = (img.width-224)/2
    bottom = (img.height-224)/2
    right = left + 224
    top = bottom + 224
    
    #crop image
    img = img.crop((left, bottom, right,    
                   top))
    
    #Normalise image and transpose colour channel
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean)/std
    img = img.transpose((2, 0, 1))
    
    
    return img
           
