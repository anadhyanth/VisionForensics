# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = "dataset"

# ============================================================
# IMAGE PARAMETERS
# ============================================================

IMAGE_SIZE = (224,224)

BATCH_SIZE = 32

EPOCHS = 25

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# ============================================================
# TRAIN GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# ============================================================
# VALIDATION GENERATOR
# ============================================================

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ============================================================
# DISPLAY CLASS LABELS
# ============================================================

print(train_generator.class_indices)

# ============================================================
# VISUALIZE IMAGES
# ============================================================

images, labels = next(train_generator)

plt.figure(figsize=(12,8))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(images[i])

    plt.axis("off")

plt.tight_layout()

plt.show()

# ============================================================
# CNN MODEL
# ============================================================

model = Sequential()

# Block 1
model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same',
        input_shape=(224,224,3)
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.25))

# Block 2

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.25))

# Block 3

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.30))

# Block 4

model.add(
    Conv2D(
        256,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        256,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.35))

# Block 5

model.add(
    Conv2D(
        512,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        512,
        (3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.40))

# Flatten

model.add(Flatten())

# Dense Layers

model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.50))

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.40))

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.30))

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

# Output

model.add(Dense(1, activation='sigmoid'))

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# ============================================================
# TRAIN MODEL
# ============================================================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Training Accuracy")

plt.legend(["Train","Validation"])

plt.show()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Training Loss")

plt.legend(["Train","Validation"])

plt.show()

# ============================================================
# EVALUATION
# ============================================================

loss, accuracy = model.evaluate(
    validation_generator
)

print("Accuracy :", accuracy)

# ============================================================
# CONFUSION MATRIX
# ============================================================

predictions = model.predict(validation_generator)

predictions = (predictions > 0.5).astype(int)

cm = confusion_matrix(
    validation_generator.classes,
    predictions[:len(validation_generator.classes)]
)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.show()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    classification_report(
        validation_generator.classes,
        predictions[:len(validation_generator.classes)]
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save("deepverify_model.h5")

# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def predict_image(image_path):

    image = load_img(
        image_path,
        target_size=(224,224)
    )

    image = img_to_array(image)

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(image)

    if prediction[0][0] > 0.5:
        print("Forged Image")
    else:
        print("Authentic Image")

# Example
# predict_image("sample.jpg")