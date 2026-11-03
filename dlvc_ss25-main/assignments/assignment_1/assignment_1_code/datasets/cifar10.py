import pickle
from typing import Tuple
import numpy as np

import os

from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        """

        # TODO implement
        # See the CIFAR-10 website on how to load the data files
        if not os.path.isdir(self.fdir):
            raise ValueError(f"{self.fdir} is not valid directory.")
        files = []
        for i in range(5):
            files.append("data_batch_"+str(i+1))
        files.append("test_batch")
        
        for f in files:
            
            if f not in os.listdir(self.fdir):
                raise ValueError(f"The file {f} is not in the directory {self.fdir}")
            
        if self.subset == Subset.TRAINING:
            batch_files = ["data_batch_"+str(i) for i in range(1,5)]
        elif self.subset == Subset.VALIDATION:
            batch_files = ["data_batch_5"]
        else:
            batch_files = ["test_batch"]

        images = []
        labels = []

        for batch_name in batch_files:
            with open(os.path.join(self.fdir,batch_name), "rb") as f:
                batch = pickle.load(f,encoding="bytes")
                data = batch[b"data"]        
                batch_labels = batch[b"labels"]

                data = data.reshape(-1,3,32,32).transpose(0,2,3,1)
                images.append(data)
                labels.extend(batch_labels)
        images = np.concatenate(images,axis=0).astype(np.uint8)
        labels = np.array(labels,dtype = np.int64)

        return images,labels


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        # TODO implement
        return len(self.labels)
        pass

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        # TODO implement
        if idx >= len(self.labels):
            raise IndexError("Index out of bounds")
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        return (image,label)
        pass

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        # TODO implement
        return len(np.unique(self.labels))
        pass
