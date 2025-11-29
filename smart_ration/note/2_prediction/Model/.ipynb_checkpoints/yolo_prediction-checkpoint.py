import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)
        
        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # Generate colors for each class at initialization
        self.colors = self.generate_colors(self.nc)
        
        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def prediction(self, image):
        row, col, d = image.shape
        
        # Convert image into square
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image
        
        # Prepare input blob for YOLO
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # Prediction
        
        # Non-Maximum Suppression (NMS)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []
        
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO
        
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:  # Confidence threshold
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                
                if class_score > 0.25:  # Class score threshold
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    
                    box = np.array([left, top, width, height])
                    
                    # Append detections
                    confidences.append(float(confidence))
                    boxes.append(box)
                    classes.append(class_id)
        
        # NMS
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        
        # Apply NMS, and check if the result is not empty
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
        if len(index) > 0:
            index = index.flatten()
            
            # Draw bounding boxes
            for ind in index:
                x, y, w, h = boxes_np[ind]
                bb_conf = int(confidences_np[ind] * 100)
                class_id = classes[ind]
                class_name = self.labels[class_id]
                color = self.colors[class_id]
                text = f'{class_name}: {bb_conf}%'
                
                # Draw rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # Create a background for the text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                cv2.rectangle(image, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)  # Background rectangle
                
                # Draw the label on top of the rectangle
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # White text

        return image  # Return the image with drawn boxes and labels
    
    def generate_colors(self, nc):
        # Generate a list of unique colors for each class
        np.random.seed(42)  # Set seed once, during initialization
        return np.random.randint(100, 255, size=(nc, 3)).tolist()
