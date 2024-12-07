import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

def startapplication():
    # Try to open webcam
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam")
        # Try alternative webcam index
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open any webcam")
            return

    print("Webcam opened successfully! Press 'q' to quit.")
    
    # Initialize model
    try:
        model = AccidentDetectionModel("model.json", 'model.weights.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        cap.release()
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        # Resize frame to expected input size (250x250)
        resized_frame = cv2.resize(frame, (250, 250))
        
        # Convert BGR to RGB and normalize
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_frame = np.expand_dims(normalized_frame, axis=0)
        
        # Get prediction
        try:
            pred, prob = model.predict_accident(input_frame)
            prob_value = prob[0][0] * 100
            
            # Draw prediction on frame
            if pred == "Accident" and prob_value > 50:
                cv2.rectangle(frame, (0, 0), (400, 50), (0, 0, 255), -1)
                text = f"Accident Detected! {prob_value:.1f}%"
                cv2.putText(frame, text, (10, 35), font, 1, (255, 255, 255), 2)
            
            # Always show the current prediction and probability
            cv2.putText(frame, f"Status: {pred} ({prob_value:.1f}%)", 
                      (10, frame.shape[0] - 20), font, 0.7, (0, 255, 0), 2)
                      
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            continue

        # Show the frame
        cv2.imshow('Accident Detection (Webcam)', frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    print("Closing application...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()