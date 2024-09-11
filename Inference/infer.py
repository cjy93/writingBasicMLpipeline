# Add the project root directory to sys.path
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.model import GarmentClassifier
from Dataset.loadData import training_loader, validation_loader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report



# Load the model architecture
model = GarmentClassifier()

# Load the saved model weights
model.load_state_dict(torch.load('./Results/model_20240911_003147_4'))

# Set the model to evaluation mode
model.eval()

# If you have a GPU available, move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Lists to store predictions and true labels
all_predictions = []
all_labels = []

# Disable gradient computation for inference
with torch.no_grad():
    for inputs, labels in validation_loader:
        # Move inputs and labels to the same device as the model
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Get predictions
        _, predictions = torch.max(outputs, 1)
        
        # Store predictions and labels
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays for easier handling
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)



# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Validation Accuracy: {accuracy:.4f}")

# Print detailed classification report
print(classification_report(all_labels, all_predictions))