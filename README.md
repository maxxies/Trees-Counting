# Aya-Trees-Counting

This project focuses on detecting and classifying trees (Palm and Non-Palm) using a Faster R-CNN model. The dataset consists of annotated images with bounding boxes that identify the location and type of trees.

## Project Structure
```
.
├── model
│   ├── model.py
├── utils
│   └── __init__.py
|   ├── config.py
|   ├── data_loader.py
|   ├── logger.py
|   ├── trainer.py
|   └── visualisations.py
├── .gitignore
├── Dockerfile
├── entrypoint.sh
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```


- **utils/**: Utility scripts, including the trainer and dataset loader.
- **models/**: Model architecture for Faster R-CNN.
- **main.py**: Main script for training and evaluating the model.
- **Dockerfile**: Docker configuration for containerized training.
- **README.md**: Project documentation.

## Metrics Used

### Mean Average Precision (mAP)
mAP is used to evaluate the performance of the object detection model. It calculates the precision of the model at different recall levels and provides an overall metric that balances precision and recall.

### Loss
The training process optimizes several loss components:
- **Box Regression Loss**: Measures how well the predicted bounding boxes align with the ground truth.
- **Classification Loss**: Measures the accuracy of the predicted class labels.
- **Objectness Loss**: Measures the confidence of the predicted objects.
- **RPN Box Loss**: Measures the accuracy of the Region Proposal Network.


## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch
- Docker (for containerized deployment)
- wandb (Weights and Biases for logging)

### Installation


