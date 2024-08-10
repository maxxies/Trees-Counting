# create a function to take only a dataloader and plot a sample of images with their bounding boxes and labels.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import transforms


def plot_sample_images(dataloader, num_images=4):
    """Plot a sample of images with their bounding boxes and labels.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the images and labels.
    """
    colors = ['r', 'g']
    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    # Get a batch of data
    for i, (img, target) in enumerate(dataloader):

        # Get the image and target
        img = img[0].permute(1, 2, 0).numpy()
        boxes = target[0]['boxes']
        labels = target[0]['labels']

        # Plot the image
        axes[i].imshow(img)

        # Plot the bounding boxes and labels
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor=colors[label.item()], facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(xmin, ymin, f"{label.item()}", color=colors[label.item()], fontsize=12)

        axes[i].axis('off')

        if i == num_images - 1:
            break

    plt.tight_layout()
    plt.show()
