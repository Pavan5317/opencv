import cv2

# Create a VideoCapture object to capture video from a camera or video file

cap = cv2.VideoCapture('Test.mp4')  # Use 0 for the default camera

# Initialize variables for motion detection
previous_frame = None
motion_threshold = 10000  # Adjust this value based on your scene and lighting conditions
#tracker = cv2.TrackerCSRT_create()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #bb=cv2.selectROI(frame,True)
    #tracker.init(frame,bb)

    if not ret:
        break  # End of video stream

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if previous_frame is not None:
        # Compute the absolute difference between the current and previous frames
        frame_diff = cv2.absdiff(previous_frame, gray_frame)

        # Threshold the difference image to identify areas of significant motion
        _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Count the number of non-zero pixels in the thresholded image
        motion_pixels = cv2.countNonZero(threshold)

        # If the number of motion pixels exceeds the threshold, theft is detected
        if motion_pixels > motion_threshold:
            print("Theft detected!")

    # Update the previous frame
    previous_frame = gray_frame

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
