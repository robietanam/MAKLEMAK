from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt


classname = ["wheelchair", "walking_frame", "crutches", "person", "push_wheelchair"]

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained model file

# Run inference on an image
image_path = "image.png"  # Replace with your image file
results = model(image_path)

# Get the output image with detections
output_image = results[0].plot()

# Convert BGR (OpenCV) to RGB (Matplotlib)
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(output_image)
plt.axis("off")  # Hide axis
plt.show()

# Export the model to ONNX format
model.export(format="onnx", dynamic=True, simplify=True)

# Load the exported ONNX model
onnx_model = YOLO("TSTAR.onnx")

# Run inference
results = onnx_model(image_path)