import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader # we had to import this because of data loading in __init__ of the class
import os
# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
        
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """

        ## TODO implement 
        # Store given variables as instance variables for use in other class methods e.g. self.model = model.
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency

        #self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        #self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


        # Track best model
        self.best_val_pcacc = 0.0  # best mean per-class accuracy so far

        # Move model to correct device
        self.model.to(self.device)

        # Initialize W&B logger
        self.wandb_logger = WandBLogger(enabled=True, model=self.model, run_name= "ResNet-Default-Train")



    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        self.model.train()  # Set the model to training mode
        self.train_metric.reset()  # Reset the training metric at the start of each epoch

        running_loss = 0.0
        num_batches = 0

        for inputs, targets in tqdm(self.train_data, desc=f"[Training] Epoch {epoch_idx}"):
            inputs = inputs.to(self.device)  # Move inputs to the device (GPU or CPU)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()  # Zero the gradients

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()  # Accumulate the loss

            num_batches += 1
            self.train_metric.update(outputs.detach().cpu(), targets.detach().cpu())

        

        average_loss = running_loss / num_batches  # Calculate average loss
        accuracy = self.train_metric.accuracy()  # Get accuracy from the metric
        per_class_accuracy = self.train_metric.per_class_accuracy()

        print("\n------- Training epoch results start -------")
        print(f"\n______epoch {epoch_idx}")
        print(f"Train loss: {average_loss:.4f}")
        print(f"Train accuracy: {accuracy:.4f}")
        print(f"Train per-class accuracy: {per_class_accuracy:.4f}")

        for class_name in self.train_metric.classes:
            total = self.train_metric.total_pred[class_name]
            if total == 0:
                class_accuracy = 0.0
            else:
                class_accuracy = 100.0 * self.train_metric.correct_pred[class_name] / total
        
            print(f"Accuracy for class: {class_name:6} is {class_accuracy:.2f}%")

        # Log metrics to W&B
        if self.wandb_logger.enabled:
            self.wandb_logger.log({
                "Train Loss": average_loss,
                "Train Accuracy": accuracy,
                "Train Per Class Accuracy": per_class_accuracy,
            })

        print("\n------- Training epoch results end -------\n")

        return average_loss, accuracy, per_class_accuracy


    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
     
        self.model.eval()  # Set model to evaluation mode
        self.val_metric.reset()

        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_data, desc=f"[Validation] Epoch {epoch_idx}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                running_loss += loss.item()
                num_batches += 1

                self.val_metric.update(outputs.cpu(), targets.cpu())

        avg_loss = running_loss / num_batches
        acc = self.val_metric.accuracy()
        pcacc = self.val_metric.per_class_accuracy()
        print("\n------- Validation epoch results start -------")
        print(f"\n______epoch {epoch_idx}")
        print(f"Validation loss: {avg_loss:.4f}")
        print(f"Validation accuracy: {acc:.4f}")
        print(f"Validation per class accuracy: {pcacc:.4f}")
        for class_name in self.val_metric.classes:
            total = self.val_metric.total_pred[class_name]
            if total == 0:
                class_acc = 0.0
            else:
                class_acc = self.val_metric.correct_pred[class_name] / total
            print(f"Accuracy for class: {class_name:6} is {class_acc:.2f}")

        # Log metrics to W&B
        if self.wandb_logger.enabled:
            self.wandb_logger.log({
                "Validation Loss": avg_loss,
                "Validation Accuracy": acc,
                "Validation Per Class Accuracy": pcacc
            })

        print("\n------- Validation epoch results end -------\n")

        return avg_loss, acc, pcacc


    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        ## TODO implement
        for epoch in tqdm(range(self.num_epochs), desc="Training Epochs"):
            train_loss, train_acc, train_pcacc = self._train_epoch(epoch)

            # Validate every val_frequency epochs
            if epoch % self.val_frequency == 0:
                val_loss, val_acc, val_pcacc = self._val_epoch(epoch)

                # Save model if this is the best validation per-class accuracy
                if val_pcacc > self.best_val_pcacc:
                    self.best_val_pcacc = val_pcacc
                    model_path = self.training_save_dir +"model_best.pth"
                    self.model.save(model_path)
                    print(f"New best model has been saved to {model_path} (PCAcc: {val_pcacc:.4f})")

        self.wandb_logger.finish()  # Finish W&B logging
        print("Training complete!")

