import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# List all the labels of the classes
class_names = [
    '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', '=', 'X', 'alpha', 'beta', 'geq', 'gt', 'lambda',
    'leq', 'lt', 'neq', 'y'
]

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network (CNN) for image classification.

    Architecture:
    1. Two convolutional layers with ReLU activation.
    2. Max pooling after each convolutional layer.
    3. Dropout for regularization.
    4. Fully connected layers for classification.

    Parameters:
    -----------
    num_classes : int
        Number of output classes for the classification task.
    """

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 11 * 11, 128)  # Input size calculated based on pooling and input size
        self.fc2 = nn.Linear(128, num_classes)   # Output layer for classification

    def forward(self, x):
        """
        Define the forward pass of the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, 1, 44, 44] (grayscale images).

        Returns:
        --------
        torch.Tensor
            Output tensor with predictions for each class.
        """
        # Convolution + ReLU + Pooling (Layer 1)
        x = self.pool(torch.relu(self.conv1(x)))  # Output shape: [batch_size, 16, 22, 22]

        # Convolution + ReLU + Pooling (Layer 2)
        x = self.pool(torch.relu(self.conv2(x)))  # Output shape: [batch_size, 32, 11, 11]

        # Flatten the feature maps for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to shape: [batch_size, 32 * 11 * 11]

        # Dropout for regularization
        x = self.dropout(x)

        # Fully connected layers with ReLU
        x = torch.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)

        return x

def load_pytorch_model(model_path="best_model.pth", num_classes=None, device="cpu"):
    """
    Load a pre-trained PyTorch model and prepare it for evaluation.

    Parameters:
    -----------
    model_path : str, optional
        Path to the model file (default: "best_model.pth").
    num_classes : int, optional
        Number of classes for the model. If None, assumes `class_names` is defined globally.
    device : str, optional
        Device to load the model onto, e.g., "cpu" or "cuda" (default: "cpu").

    Returns:
    --------
    torch.nn.Module
        The loaded and evaluated PyTorch model.
    """

    # Determine the number of classes
    if num_classes is None:
        if 'class_names' not in globals():
            raise ValueError("num_classes is None, and 'class_names' is not defined globally.")
        num_classes = len(class_names)

    # Initialize the model
    model = SimpleCNN(num_classes=num_classes)

    # Load the model state from the file
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")

    # Set the model to evaluation mode and move it to the specified device
    model.eval()
    model.to(device)

    return model


def predict_images_pytorch(images, model_path="best_model.pth"):
    """
    Predict the class of multiple images using a PyTorch model.

    Parameters:
    -----------
    images : list of np.ndarray
        List of images as NumPy arrays (grayscale or color).
    model_path : str, optional
        Path to the pre-trained PyTorch model (default: "best_model.pth").

    Returns:
    --------
    list of dict
        List of predictions, each containing:
        - 'class_idx': Predicted class index.
        - 'class_name': Predicted class name.
        - 'probability': Confidence of the prediction.
    """

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_pytorch_model(model_path, num_classes=len(class_names), device=device)

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    results = []  # Store predictions
    # plt.figure(figsize=(8, 2 * len(images)))

    for i, img_np in enumerate(images):
        # Convert color images to grayscale if necessary
        if img_np.ndim == 3 and img_np.shape[-1] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_np

        # Apply transformations and prepare input tensor
        try:
            input_tensor = transform(img_gray).unsqueeze(0).to(device)
        except Exception as e:
            raise ValueError(f"Error processing image {i + 1}: {e}")

        # Perform prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            conf, predicted_class = torch.max(probabilities, 1)
            conf = conf.item()
            predicted_idx = predicted_class.item()

        # Get predicted label
        predicted_label = class_names[predicted_idx]

        # Append result
        results.append({
            'class_idx': predicted_idx,
            'class_name': predicted_label,
            'probability': conf
        })

        # Visualization
        # plt.subplot(len(images), 2, 2 * i + 1)
        # plt.imshow(img_gray, cmap='gray')
        # plt.title(f"Image {i + 1}")
        # plt.axis('off')

        # plt.subplot(len(images), 2, 2 * i + 2)
        # plt.text(
        #     0.5, 0.5,
        #     f"Predicted: {predicted_label}\nConfidence: {conf * 100:.2f}%",
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=plt.gca().transAxes,
        #     fontsize=12
        # )
        # plt.axis('off')

    # plt.tight_layout()
    # plt.show()
    return results
