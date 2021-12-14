import os

#Deep learning package
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


#Creating a custom dataset for augmented images
class HumanImageDatasetAugmented(Dataset):
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


#Creating the datset by creating the class object
def create_dataset_augmented(files, labels):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    transform_augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.8),
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.Normalize(mean, std)
    ])

    human_image_dataset_augmented = HumanImageDatasetAugmented(image_name=files, image_label=labels, 
                                                            transform=transform_augmentation)
    
    return human_image_dataset_augmented


#Splitting the data into train, validation and test set
def split_data_cnn_augmented(dataset):
    lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]

    train_augmented, test_augmented = torch.utils.data.random_split(dataset, lengths)

    lengths = [int(len(train_augmented)*0.8), int(len(train_augmented)*0.2)]

    train_augmented, valid_augmented = torch.utils.data.random_split(train_augmented, lengths)

    return train_augmented, valid_augmented, test_augmented


#Creating a dataloader using the split data
def dataloader_augmented(train, valid, test):
    BATCH_SIZE=32

    train_dataloader_augmented = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader_augmented = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader_augmented = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    print('Training sample size: ', len(train_dataloader_augmented.dataset), \
        'Validation dataset size: ', len(valid_dataloader_augmented.dataset), \
        'Test sample size: ', len(test_dataloader_augmented.dataset))

    return train_dataloader_augmented, valid_dataloader_augmented, test_dataloader_augmented