from torchvision.models import detection

class Model:
    """Class to load Faster R-CNN model for fine-tuning."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """Initialize the model with the number of classes."""
        self.model = detection.fasterrcnn_resnet50_fpn(weights=pretrained)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

    def get_model(self):
        """Return the actual PyTorch model."""
        return self.model