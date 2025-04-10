import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model with CPU provider
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# YOLOv8 expects 640x640 input
input_size = 640

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize, normalize, and reshape the frame
    img = cv2.resize(frame, (input_size, input_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # CHW
    input_tensor = np.expand_dims(img_transposed, axis=0)

    # Inference
    outputs = session.run(None, {input_name: input_tensor})
    detections = outputs[0][0]  # Shape: (num_detections, 6)

    # Post-processing and drawing boxes
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:
            # Scale boxes back to original frame size
            x_scale = frame.shape[1] / input_size
            y_scale = frame.shape[0] / input_size
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{int(cls)}:{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show the result
    cv2.imshow("ONNX Webcam Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
