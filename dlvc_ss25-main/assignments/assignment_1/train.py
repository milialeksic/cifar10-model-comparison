## Feel free to change the imports according to your implementation and needs
import argparse
import os



import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os
from torch.utils.data import DataLoader # added import for DataLoader


from assignment_1_code.models.class_model import (
    DeepClassifier,
)  # etc. change to your model
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from torchvision.models import resnet18  # added model

def train(args):

    ### Implement this function so that it trains a specific model as described in the instruction.md file
    ## feel free to change the code snippets given here, they are just to give you an initial structure
    ## but do not have to be used if you want to do it differently
    ## For device handling you can take a look at pytorch documentation

    # define the batch size
    batch_size = 128

    # train_transform = v2.Compose(
    #     [
    #         v2.ToImage(),
    #         v2.ToDtype(torch.float32, scale=True),
    #         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    weight_decay = 0.01
    lr = 0.001
    train_transforms = v2.Compose([
        v2.ToPILImage(),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    regularization_mode = args.regularization_mode
    if regularization_mode == "augmentation":
        train_transforms = v2.Compose([
            v2.ToPILImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32,padding = 4),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif regularization_mode == "weight_decay":

        weight_decay = 0.05
    elif regularization_mode == "learning_rate":
        lr = 0.0005
    elif regularization_mode == "augmentation_weight_decay":
        train_transforms = v2.Compose([
            v2.ToPILImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32,padding = 4),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        weight_decay = 0.05
    elif regularization_mode == "augmentation_learning_rate":
        train_transforms = v2.Compose([
            v2.ToPILImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32,padding = 4),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        lr = 0.0005
    elif regularization_mode == "augmentation_weight_decay_learning_rate":
        train_transforms = v2.Compose([
            v2.ToPILImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32,padding = 4),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        lr = 0.0005
        weight_decay = 0.05
    

    cifar10_path = args.data_dir

    train_data = CIFAR10Dataset(fdir=cifar10_path,subset=Subset.TRAINING,transform=train_transforms)
    val_data = CIFAR10Dataset(fdir=cifar10_path,subset=Subset.VALIDATION,transform=val_transform)

    # pass loaders because in trainer.py we don't have DataLoader class as import
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


    # print(len(train_data))
    # print(len(val_data))
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = resnet18(weights=None) 
    backbone.fc = torch.nn.Linear(backbone.fc.in_features, 10) # we need to addapt because CIFAR10 has 10 classes
    model = DeepClassifier(backbone)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    # model_save_dir = Path("saved_models")
    # model_save_dir.mkdir(exist_ok=True)
    save_path = Path("./saved_models/resnet_model") / regularization_mode
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    trainer = ImgClassificationTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data_loader,
        val_data_loader,
        device,
        args.num_epochs,
        "./saved_models/resnet_model/"+regularization_mode+"/",
        batch_size=batch_size,  # feel free to change
        val_frequency=val_frequency,
    )
    trainer.train()


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Testing")
    # args.add_argument(
    #     "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    # )
    args.add_argument(
        "--data_dir",type = str, required= True, help = "Path to the CIFAR-10 data directory"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) # this goes before import torch because it already recognized the device
    # args.gpu_id = str(args.gpu_id)
    args.num_epochs = 30
    # args.regularization_mode = "default"
    # train(args)

    # args.regularization_mode = "augmentation"
    # train(args)
    
    # args.regularization_mode = "weight_decay"
    # train(args)
   
    args.regularization_mode = "learning_rate"
    train(args)
   
    args.regularization_mode = "augmentation_weight_decay"
    train(args)
   
    args.regularization_mode = "augmentation_learning_rate"
    train(args)
   
    args.regularization_mode = "augmentation_weight_decay_learning_rate"
    train(args)
    
    
