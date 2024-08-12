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

### Training the Model

1. Clone the repository:
```bash
git clone <repo_url>
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set directory paths and epochs:
```python   
DATA_DIR = "<your_data_directory>"
EPOCHS = "<number_of_epochs>"
```

4. Set up Weights and Biases for logging:
```bash
wandb login
```

5. Run the training script:
```python
import main

test_data, predictions, target = main.run(DATA_DIR, EPOCHS)

6. Evaluate the model:
```python
from utils.visualisations import plot_sample_images

plot_comparison_images(images, target, predictions,num_images=10)
```


### Training with Docker
1. Pull the Docker image:
```bash
docker pull <image_name>
```

2. Run the Docker container:
```bash
docker run -it --gpus all \
    -v /path/to/your/data:/app/data \
    -v /path/to/save/output:/app/output \
    -e EPOCHS=50 \
    -e WANDB_API_KEY=your_wandb_api_key \
    palm-tree-counter
```

## Model Monitoring with Weights and Biases
Weights and Biases was used to monitor the training process and log metrics. You can view the project dashboard [here](https://wandb.ai/ahiamadzormaxwell7/counting-trees).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



