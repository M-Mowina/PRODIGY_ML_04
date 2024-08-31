import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('hand_gesture_V2.h5')

# Define the label mapping
label_mapping = {
    0: '01_palm',
    1: '02_l',
    2: '03_fist',
    3: '04_fist_moved',
    4: '05_thumb',
    5: '06_index',
    6: '07_ok',
    7: '08_palm_moved',
    8: '09_c',
    9: '10_down'
}

# Define a function to preprocess the image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img_array = np.array(img)
    img_array = img_array.reshape(-1, 64, 64, 1)
    img_array = img_array / 255.0
    return img_array

# Open the IP Webcam stream
cap = cv2.VideoCapture('http://192.168.43.5:4747/video')

# Set the desired camera window size (width, height)
window_width = 800
window_height = 600

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to the desired size
    frame = cv2.resize(frame, (window_width, window_height))

    # Preprocess the frame
    img_array = preprocess_image(frame)

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = label_mapping.get(predicted_class, "Unknown")

    # Display the predicted label on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Camera Feed', frame)


    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
