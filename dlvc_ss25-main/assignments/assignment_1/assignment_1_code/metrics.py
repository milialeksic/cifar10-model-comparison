from abc import ABCMeta, abstractmethod
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        
        ## TODO implement
        
        if len(prediction.shape) != 2:
            raise ValueError(f"Expected prediction should have 2 dimensions, got {prediction.ndim}")
        if len(target.shape) != 1:
            raise ValueError(f"Expected target should have 1 dimension, got {target.ndim}")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(f"Expected prediction and target should have same batch size, got {prediction.shape[0]} and {target.shape[0]}")

        pred_classes = torch.argmax(prediction, dim=1)  
        
        for pred, true in zip(pred_classes, target):
            true_label = self.classes[true.item()]
            pred_label = self.classes[pred.item()]

            self.total_pred[true_label] += 1
            if pred == true:
                self.correct_pred[true_label] += 1
                self.n_matching += 1

            self.n_total += 1


    def __str__(self):
        """
        Return a string representation of the performance, accuracy and per class accuracy.
        """

        ## TODO implement
        if self.n_total == 0:
            return "No predictions made."

        # Overall accuracy
        overall_acc = 100.0 * self.n_matching / self.n_total
        result = [f"Accuracy: {overall_acc:.2f}%"]

        # Per-class accuracy
        for classname in self.classes:
            total = self.total_pred[classname]
            if total == 0:
                acc = 0.0  # Avoid division by zero
            else:
                acc = 100.0 * self.correct_pred[classname] / total
            result.append(f"Class '{classname}': {acc:.2f}%")

        return "\n".join(result)

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        ## TODO implement
        if self.n_total == 0:
            return 0.0
        
        return self.n_matching / self.n_total



    def per_class_accuracy(self) -> float:
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        class_accuracies = []

        for classname in self.classes:
            total = self.total_pred[classname]
            correct = self.correct_pred[classname]

            if total == 0:
                acc = 0.0  
            else:
                acc = correct / total

            class_accuracies.append(acc)

        if len(class_accuracies) == 0: # avoid division by 0
            return 0.0

        return sum(class_accuracies) / len(class_accuracies)
