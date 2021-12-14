import os

#Deep learning package
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

#Creating a custom dataset for compressed images
class HumanImageDatasetCompressed(Dataset):
    def __init__(self, image_name, image_label, transform=None):
        self.image_name = image_name
        self.image_label = image_label
        self.transform = transform

    def __len__(self):
        return len(self.image_label)

    def __getitem__(self, index):
        image_path = os.path.join('./Compressed Images/', str(self.image_name[index]))
        image = read_image(image_path, mode=ImageReadMode.RGB)
        label = self.image_label[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


#Creating a custom dataset
class HumanImageDataset(Dataset):
    def __init__(self, image_name, image_label, transform=None):
        self.image_name = image_name
        self.image_label = image_label
        self.transform = transform

    def __len__(self):
        return len(self.image_label)

    def __getitem__(self, index):
        image_path = os.path.join('./images/', str(self.image_name[index]))
        image = read_image(image_path, mode=ImageReadMode.RGB)
        label = self.image_label[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
        

#Model of a Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=32*16*3*3*3*3*3, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=7)
        )

    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



#Model of a Neural Network
class NeuralNetworkKBest(nn.Module):
    def __init__(self):
        super(NeuralNetworkKBest, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=32*17*13*2, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=7)
        )

    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


#Method to calculate the time of an epoch
def epoch_duration(start, end):
    time = end - start
    time_min = int(time / 60)
    time_sec = int(time - (time_min * 60))
    return time_min, time_sec



#Create the dataset for compressed class
def create_dataset_compressed(files, labels):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transform_no_augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    human_image_dataset = HumanImageDatasetCompressed(image_name=files, image_label=labels, transform=transform_no_augmentation)
    return human_image_dataset


#Creating the dataset by creating the class object
def create_dataset(files, labels):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transform_no_augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    human_image_dataset = HumanImageDataset(image_name=files, image_label=labels, transform=transform_no_augmentation)
    return human_image_dataset


#Splitting the data into training, validation and test dataset
def split_data_cnn(dataset):
    lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]

    train, test = torch.utils.data.random_split(dataset, lengths)

    lengths = [int(len(train)*0.8), int(len(train)*0.2)]

    train, valid = torch.utils.data.random_split(train, lengths)

    return train, valid, test


#Creating the Dataloader that will load the data to the Deep Learning model
def dataloader(train, valid, test):
    BATCH_SIZE = 32

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    print('Training sample size: ', len(train_dataloader.dataset), \
        'Validation dataset size: ', len(valid_dataloader.dataset), \
        'Test sample size: ', len(test_dataloader.dataset))  

    return train_dataloader, valid_dataloader, test_dataloader



  