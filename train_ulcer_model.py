import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset path
dataset_path = "dataset"

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path, target_size=(224, 224), batch_size=4, class_mode='categorical', subset='training')

val_generator = datagen.flow_from_directory(
    dataset_path, target_size=(224, 224), batch_size=4, class_mode='categorical', subset='validation')

# Debugging: Check detected class labels
print("Class Labels:", train_generator.class_indices)

# Ensure the number of detected classes matches model output
num_classes = len(train_generator.class_indices)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Match detected classes
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Save Model to /models folder
model.save("models/ulcer_model.h5")
print("Model saved successfully in /models folder!")
