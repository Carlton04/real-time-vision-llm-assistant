
# Import necessary modules
from ultralytics import YOLO
import torch

# Check if MPS (Metal Performance Shaders) is available for MacBook M1 Pro
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the pre-trained YOLO model (yolov8m.pt, for example)
model = YOLO("yolov10m.pt")  # Replace with "yolov10m.pt" if available or preferred

# Define dataset configuration file
data_yaml = "SKU-110K.yaml"

# Train the model on custom data
results = model.train(data=data_yaml,        # Path to dataset YAML
                      epochs=20,            # Set number of epochs
                      imgsz=640,            # Set image size (adjust if necessary)
                      batch=8,             # Batch size (adjust based on memory)
                      device=device,        # Use MPS (Apple GPU) or CPU
                      save=True,            # Save training results
                      project="runs",       # Save results in the 'runs' folder
                      name="custom_training",  # Folder for this training run
                      workers=2,            # Dataloader workers (tune for speed)
                      plots=True            # Save plots (confusion matrix, loss graphs, etc.)
                     )

# Save the trained weights as customsku110k.pt in the runs/custom_training/weights folder
model_path = "runs/custom_training/weights/YOLOV10mcustomsku110k.pt"
model.save(model_path)

# After training, visualize some results (optional)
results.show()

# Visualize confusion matrix and other performance metrics
results.plot()

# You can also export the model to other formats if needed
# For example, export to ONNX format:
# model.export(export_format="onnx")
