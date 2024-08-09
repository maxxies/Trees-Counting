# class to load faster r cnn model for fine tuning
from torchvision.models import detection

class Model:
    """Class to load Faster R-CNN model for fine-tuning.

    Attributes:
        model (torchvision.models.detection.fasterrcnn_resnet50_fpn): The
            Faster R-CNN model.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        """Initialize the model with the number of classes.

        Args:
            num_classes (int): The number of classes in the dataset.
            pretrained (bool, optional): Whether to load the pretrained weights.
                Defaults to True.
        """
        self.model = detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained
        )

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

    def __call__(self, *args, **kwargs):
        """Forward pass of the model."""
        return self.model(*args, **kwargs)        
       