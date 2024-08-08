import pandas as pd
import os
import torch
from torch.utils.data import Dataset 
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class TrainingDataSet(Dataset):
    """Dataset class for loading train images from a folder structure.

    Attributes:
        data_dir (str): The directory where the image folders are located.
        annotation_df (Dataframe): Dataframe of annotations of the images loaded.
        transform (transforms.Compose): Transformations to be applied to the images.
        imgs (list): List of paths of images in the data directory.
    """
    def __init__(self, data_dir, annotation, image_transform):
        """Initializes the dataset class with the given parameters.

        Args:
            data_dir (str): The directory where the image folders are located.
            annotation (str): The path to the annotations of the images.
            image_transform (transforms.Compose): Transformations to be applied to the images.
        """
        self.data_dir = data_dir
        self.annotation_df = pd.read_csv(annotation)
        self.transform = image_transform
        self.imgs = sorted(os.listdir(self.data_dir))    # load all image files, sorting them to ensure that they are aligned


    def __getitem__(self, idx):
        """Samples an image and return the transformed image and an object containing bounding box and transformed label.

        Args:
            idx (int): Index used for sampling an image and their label

        Returns:
            img (Tensor): Transformed image.
            target (dict): Object containing the bounding box details and label of the transformed image.
        """
        # load all images 
        img_path = os.path.join(self.data_dir, self.imgs[idx])
        bbox = (self.annotation_df[self.annotation_df['img_id'] == self.imgs[idx]][['xmin','ymin','xmax','ymax']]).values[0]
        label = [1]
        label = torch.tensor(labels, dtype=torch.int64)

        img = read_image(img_path)

        # Wrap sample and targets into torchvision tv_tensors
        img = tv_tensors.Image(img)

        # Create a target dictionary containing the bounding boxes and labels
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(bbox, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __matrix__(self,idx):
        img_path = os.path.join(self.img_paths, self.imgs[idx])
        img = read_image(img_path)

        # Return the dimensions of the image (height, width, number of channels)
        return len(img),len(img[0]),len(img[0][0])