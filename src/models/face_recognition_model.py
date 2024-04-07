import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D #type: ignore
from tensorflow.keras.models import Model #type: ignore

def build_and_train_model(train_dir, val_dir, num_classes, image_size=(224, 224), batch_size=32, epochs=10):

    # Define data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load MobileNetV2 base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Prepare data generators
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical')
    val_generator = val_datagen.flow_from_directory(val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical')
    
    # Train the model
    model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=epochs,
              validation_data=val_generator, validation_steps=val_generator.samples // batch_size)
    
    # Save the trained model
    model.save('custom_face_recognition_model.h5')