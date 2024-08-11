import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.config import Config
import os

def plot_sample_images(dataloader, output_dir, num_images=4):
    """Plot a sample of images with their bounding boxes and labels.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the images and labels.
    """
    colors = ['r', 'b']
    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    # Get a batch of data
    for i, (img, target) in enumerate(dataloader):
        # Move image and target to CPU to avoid GPU memory issues
        img = img[0].cpu().permute(1, 2, 0).numpy()
        boxes = target[0]['boxes'].cpu()
        labels = target[0]['labels'].cpu()

        # Plot the image
        axes[i].imshow(img)

        # Plot the bounding boxes and labels
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor=colors[label.item() - 1], facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(xmin, ymin, f"{label.item()}", color=colors[label.item() - 1], fontsize=12)

        axes[i].axis('off')

        if i == num_images - 1:
            break

    plt.savefig(os.path.join(output_dir, "sample_images.png"))
    plt.close()
    # plt.tight_layout()
    # plt.show()



def plot_comparison_images(images, targets, predictions, output_dir, confidence = 0.7, num_images=4):
    """Plot images with ground truth and predicted bounding boxes side by side.

    Args:
        images (List[torch.Tensor]): List of image tensors.
        targets (List[Dict]): List of ground truth targets.
        predictions (List[Dict]): List of model predictions.
        num_images (int, optional): Number of images to plot. Defaults to 4.
    """
    colors = ['r', 'b']
    fig, axes = plt.subplots(num_images, 2, figsize=(15, 15 * num_images // 2))

    if num_images == 1:
        axes = [axes]  

    for i in range(num_images):
        img = images[i].cpu().permute(1, 2, 0).numpy()

        # Plot ground truth
        axes[i][0].imshow(img)
        for box, label in zip(targets[i]['boxes'].cpu(), targets[i]['labels'].cpu()):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=colors[label.item() - 1], facecolor='none')
            axes[i][0].add_patch(rect)
            axes[i][0].text(xmin, ymin, f"{label.item()}", color=colors[label.item() - 1], fontsize=12)
        axes[i][0].set_title("Ground Truth")
        axes[i][0].axis('off')

        # Plot predictions
        axes[i][1].imshow(img)
        for box, label, score in zip(predictions[i]['boxes'].cpu(), predictions[i]['labels'].cpu(), predictions[i]['scores'].cpu()):
            if score < confidence:
                continue
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=colors[label.item() - 1], facecolor='none')
            axes[i][1].add_patch(rect)
            axes[i][1].text(xmin, ymin, f"{label.item()}", color=colors[label.item() - 1], fontsize=12)
        axes[i][1].set_title("Predictions")
        axes[i][1].axis('off')

    plt.savefig(os.path.join(output_dir, "predictions.png"))
    plt.close()
