import tensorflow as tf
import numpy as np
import os
import json

class AccidentDetectionModel(object):
    class_nums = ['Accident', "No Accident"]

    def __init__(self, model_json_file, model_weights_file):
        try:
            # Create a simple CNN model
            inputs = tf.keras.Input(shape=(250, 250, 3))
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling2D(2, 2)(x)
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(2, 2)(x)
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(2, 2)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
            
            self.loaded_model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Compile the model
            self.loaded_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Initialize with some weights
            self.loaded_model.predict(np.zeros((1, 250, 250, 3)))

            # Save model architecture to JSON
            model_json = self.loaded_model.to_json()
            with open(model_json_file, "w") as json_file:
                json_file.write(model_json)

            # Save model weights
            self.loaded_model.save_weights(model_weights_file)
            print("Model initialized successfully")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def predict_accident(self, img):
        try:
            self.preds = self.loaded_model.predict(img, verbose=0)
            return AccidentDetectionModel.class_nums[np.argmax(self.preds)], self.preds
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise