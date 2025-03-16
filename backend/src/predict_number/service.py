import base64
from io import BytesIO
import os
from training_numbers.service import SimpleNN, MacrosNN

# from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class PredictNumber(MacrosNN):

    def __init__(self):
        self.model = SimpleNN()
        # Load the saved state dictionary
        self.model_trained_path = os.path.join(
            os.path.abspath("predict_number"), self._MNIST_MODEL_TRAINED_PATH
        )
        self.model.load_state_dict(torch.load(self.model_trained_path))

    # Function to preprocess the image
    def _preprocess_image(self, image_b64):
        # Open the image and convert to grayscale
        image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("L")

        # Resize to 28x28 (MNIST size) and convert to tensor
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),  # Use MNIST normalization
            ]
        )
        image = transform(image).unsqueeze(0)  # Add batch dimension

        return image

    # Function to predict the digit and get confidence percentage
    def predict_digit(self, image_b64):
        # Preprocess the image
        image = self._preprocess_image(image_b64)

        result = {}
        with torch.no_grad():
            output = self.model(image)
            _, predicted_digit = torch.max(output, 1)
            probabilities = F.softmax(
                output, dim=1
            )  # Apply softmax to get probabilities
            confidence, _ = torch.max(probabilities, 1)
            result.update(
                {
                    self.PREDICTED_DIGIT_KEY: predicted_digit.item(),
                    self.CONFIDENCE_KEY: confidence.item() * 100,
                }
            )

        # image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("L")
        # plt.imshow(image, cmap="gray")
        # plt.title(f"Predicted Digit: {predicted.item()}")
        # plt.axis("off")
        # plt.show()
        # print(result)
        return result
