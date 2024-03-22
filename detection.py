import cv2
import numpy as np

# Initialize the object detection model (for example, YOLO)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Function to detect objects in a frame


# Open the video capture device
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
frame_id = 0
while True:
    # Read a frame from the video capture
    _, frame = cap.read()
    frame_id +=50
    height, width, channels = frame.shape  

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0))

    net.setInput(blob)
    outs = net.forward(output_layers)   

    # If objects are detected, initialize the tracker with the first detected box
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id  = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]* width)
                center_y=int(detection[1]* height)
                w=int(detection[2]* width)
                h=int(detection[3]* height)

                #Rectangle coordinates
                x=int(center_x - w / 2)
                y=int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence=confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the frame with tracking information
    cv2.imshow('Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
