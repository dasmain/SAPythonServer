import cv2
import numpy as np

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('trained_model/deploy.prototxt', 'trained_model/mobilenet_iter_73000.caffemodel')

def niqab_detection(image):
    img = cv2.imread(image)

    (h, w) = img.shape[:2]

    # Preprocess the image: resize and mean subtraction
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (180, 500)), 0.007843, (300, 300), 0)

    # Set the input to the pre-trained deep learning network
    net.setInput(blob)

    # Perform forward pass to get the output
    detections = net.forward()
    eye_class_ids = [ 15]
    num_faces = 0

    for i in range(detections.shape[2]):
    # Extract the confidence (i.e., probability) associated with the detection
    
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.55:  # Adjust this threshold as needed
            # Extract the index of the class label from the 'detections', which is in the 1st element
            idx = int(detections[0, 0, i, 1])
            # print(f"Detected class ID: {idx} with confidence: {confidence}")
            # If the detection is an eye
            if idx in eye_class_ids:
                num_faces += 1 
                # Compute the (x, y)-coordinates of the bounding box for the object
                # box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # (startX, startY, endX, endY) = box.astype("int")
                
                # Draw the bounding box around the detected eye
                # cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # Display the output image
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return num_faces