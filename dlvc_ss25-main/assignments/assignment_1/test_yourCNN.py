import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models import resnet18  # change to the model you want to test
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.wandb_logger import WandBLogger
from assignment_1_code.models.cnn import YourCNN



def test(args, wandb_run_name):
    # batch_size
    batch_size = 128
    dropout = 0.1

    running_loss = 0.0
    num_batches = 0



    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_data = CIFAR10Dataset(fdir=args.data_dir, subset=Subset.TEST, transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.current_device()}")


    cnn_model = YourCNN(dropout=dropout)
    model = DeepClassifier(cnn_model)
    model.load(args.path_to_trained_model)
    model.to(device=device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    test_metric = Accuracy(classes=test_data.classes)

    wandb_logger = WandBLogger(enabled=True, model=model, run_name=wandb_run_name)

    with torch.no_grad():
        for inputs, targets in tqdm(test_data_loader, desc=f"[Testing] Combination: {wandb_run_name} "):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            running_loss += loss.item()
            num_batches += 1 

            test_metric.update(outputs.cpu(), targets.cpu())

            
            
    avg_loss = running_loss / num_batches
    acc = test_metric.accuracy()
    pcacc = test_metric.per_class_accuracy()

    result_string = "\n----- Test Results -----\n" + \
                "Test Loss: " + format(avg_loss, ".4f") + "\n\n" + \
                "Test Accuracy: " + format(acc, ".4f") + "\n" + \
                "Test Per-Class Accuracy: " + format(pcacc, ".4f") + "\n"


    print(result_string)

    wandb_logger.log({
        "Test Loss": avg_loss,
        "Test Accuracy": acc,
        "Test Per Class Accuracy": pcacc,
    })

    for class_name in test_metric.classes:
        total = test_metric.total_pred[class_name]
        if total == 0:
            class_acc = 0.0
        else:
            class_acc = 100 * test_metric.correct_pred[class_name] / total
        print(f"Accuracy for class: {class_name:6} is {class_acc:.2f}")
        result_string += "Accuracy for class: " + format(class_name, "<6") + " is " + format(class_acc, ".2f") + "\n"

    result_folder_path = os.path.dirname(args.path_to_trained_model)
    result_file_path = os.path.join(result_folder_path, "test_result.txt")

    with open(result_file_path, 'w') as file:
        file.write(result_string)

if __name__=="__main__":
    args = argparse.ArgumentParser(description="Testing")
    args.add_argument(
        "--data_dir",type=str,required=True,help="Path to CIFAR-10 data"
    )

    if not isinstance(args,tuple):
        args = args.parse_args()

    optm_method_types = ['default', "augmentation", "weight_decay", "learning_rate", "augmentation_weight_decay", "augmentation_learning_rate", "augmentation_weight_decay_learning_rate"]
    
    for type in optm_method_types:
        args.path_to_trained_model = f"saved_models/your_cnn_model/{type}/model_best.pth"

        # formating for wandb run name
        type_parsed = type.replace('_', '-')
        type_parsed = type_parsed.capitalize()
        wandb_run_name = "YourCNN-" + type_parsed + "-Test"

        test(args=args, wandb_run_name=wandb_run_name)
        