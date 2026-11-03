import argparse
import torch
from pathlib import Path
import os
from torchvision import transforms
import torchvision.transforms.v2 as v2
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.cnn import YourCNN
from torch.utils.data import DataLoader


def train(cifar10_path, regularization_mode):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available device: ", device)
    weight_decay = 0.01
    lr = 0.001
    num_epochs = 30
    batch_size = 128
    dropout = 0.1

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if regularization_mode == "augmentation":
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding = 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif regularization_mode == "weight_decay":

        weight_decay = 0.05
    elif regularization_mode == "learning_rate":
        lr = 0.0005
    elif regularization_mode == "augmentation_weight_decay":
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding = 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        weight_decay = 0.05
    elif regularization_mode == "augmentation_learning_rate":
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding = 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        lr = 0.0005
    elif regularization_mode == "augmentation_weight_decay_learning_rate":
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding = 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        lr = 0.0005
        weight_decay = 0.05
    
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = CIFAR10Dataset(fdir=cifar10_path,subset=Subset.TRAINING,transform=train_transforms)
    val_data = CIFAR10Dataset(fdir=cifar10_path,subset=Subset.VALIDATION,transform=val_transforms)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    your_cnn_model = YourCNN(dropout=dropout)
    model = DeepClassifier(your_cnn_model)
    model.to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    save_path = "./saved_models/your_cnn_model/" + regularization_mode + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)

    trainer = ImgClassificationTrainer(model=model,
                                       optimizer=optimizer, 
                                       loss_fn=loss_fn, 
                                       lr_scheduler=scheduler, 
                                       train_metric=train_metric, 
                                       val_metric=val_metric, 
                                       train_data=train_data_loader, 
                                       val_data=val_data_loader, 
                                       device=device, 
                                       num_epochs=num_epochs, 
                                       training_save_dir=save_path, 
                                       batch_size=batch_size,
                                       val_frequency=5)
    
    trainer.train()

if __name__=="__main__":
    args = argparse.ArgumentParser(description="Testing")
    args.add_argument(
        "--data_dir",type = str, required= True, help = "Path to the CIFAR-10 data directory"
    )
    if not isinstance(args, tuple):
        args = args.parse_args()

    #train(args.data_dir,regularization_mode = "default")
    #train(args.data_dir,regularization_mode = "augmentation")
    #train(args.data_dir,regularization_mode = "weight_decay")
    #train(args.data_dir,regularization_mode = "learning_rate")
    #train(args.data_dir,regularization_mode = "augmentation_weight_decay")
    #train(args.data_dir,regularization_mode = "augmentation_learning_rate")
    train(args.data_dir,regularization_mode = "augmentation_weight_decay_learning_rate")

