import pandas as pd
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision import transforms
from torchvision.transforms.v2 import functional as F


class TreeDataset(Dataset):
    """Dataset class for loading train images from a folder structure.

    Attributes:
        data_dir (str): The directory where the image folders are located.
        annotation_df (DataFrame): DataFrame of annotations of the images loaded.
        transform (transforms.Compose): Transformations to be applied to the images.
        imgs (list): List of paths of images in the data directory.
    """
    def __init__(self, data_dir: str, annotation: str, image_transform: transforms.Compose | None = None):
        """Initializes the dataset class with the given parameters.

        Args:
            data_dir (str): The directory where the image folders are located.
            annotation (str): The path to the annotations of the images.
            image_transform (transforms.Compose, optional): Transformations to be applied to the images.
        """
        self.data_dir = data_dir
        self.annotation_df = pd.read_csv(annotation)
        self.transform = image_transform
        self.imgs = sorted(os.listdir(self.data_dir))  # Load all image files, sorting them to ensure they are aligned
        self.filter_dataset()


    def __getitem__(self, idx):
        """Samples an image and returns the transformed image and an object containing bounding box and transformed label.

        Args:
            idx (int): Index used for sampling an image and their label.

        Returns:
            img (Tensor): Transformed image.
            target (dict): Object containing the bounding box details and label of the transformed image.
        """
        # Load image path
        img_path = os.path.join(self.data_dir, self.imgs[idx])

        # Read the image
        img = read_image(img_path)


        # Get bounding box and label for the current image
        annotations = self.annotation_df[self.annotation_df['filename'] == self.imgs[idx]]
        bboxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values
        bboxes =  tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=F.get_size(img))
        labels = annotations['class'].apply(lambda x: 1 if x == 'Palm' else 2).values
        labels = torch.tensor(labels, dtype=torch.int64)

        # Wrap image into torchvision tv_tensors
        img = tv_tensors.Image(img)

        # Create a target dictionary to contain the bounding boxes and labels
        target = {
            "boxes": bboxes,
            "labels": labels
        }

        # Apply transformations if available
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        """Returns the total size of the data."""
        return len(self.imgs)
    
    def __image__(self, idx):
        """Returns the image without any transformations."""
        img_path = os.path.join(self.data_dir, self.imgs[idx])
        img = cv2.imread(img_path)

        return img
    
    def filter_dataset(self):
        """Filter the dataset by removing images with no labels and incorrect bounding boxes."""
        self.annotation_df = self.annotation_df[self.annotation_df['class'].notnull()]
        self.annotation_df = self.annotation_df[self.annotation_df['xmin'] < self.annotation_df['xmax']]
        self.annotation_df = self.annotation_df[self.annotation_df['ymin'] < self.annotation_df['ymax']]

        # Filter out images with no annotations
        valid_filenames = self.annotation_df['filename'].unique()
        self.imgs = [img for img in self.imgs if img in valid_filenames]

    def get_transform(train: bool):
        """Returns the transformations to be applied to the images.

        Args:
            train (bool): If True, apply training transformations. Otherwise, apply test transformations.

        Returns:
            transforms.Compose: The transformations to be applied to the images.
        """
        transforms = []

        # If training, add a random horizontal flip transformation with a probability of 0.5
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))

        # Convert the image to a float tensor and scale image
        transforms.append(T.ToDtype(torch.float, scale=True))

        # Convert the image to a pure tensor
        transforms.append(T.ToPureTensor())

        return T.Compose(transforms)
    
    def collate_fn(batch):
        """Collate function to be used in the DataLoader.

        Args:
            batch (list): List of tuples containing the image and target.

        Returns:
            list: List of images and targets.
        """
        return list(zip(*batch))