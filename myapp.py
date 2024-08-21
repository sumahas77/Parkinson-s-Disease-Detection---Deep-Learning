import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

def visualize_images():
    categories = ['healthy', 'parkinson']
    for category in categories:
        plt.figure(figsize=(12, 12))
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            img = load_img(f"drawings/spiral/training/{category}/" + os.listdir(f"drawings/spiral/training/{category}")[i])
            plt.imshow(img)
        plt.show()

visualize_images()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

spiral_train_generator = train_datagen.flow_from_directory(
    'drawings/spiral/training',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

spiral_test_generator = test_datagen.flow_from_directory(
    'drawings/spiral/testing',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

wave_train_generator = train_datagen.flow_from_directory(
    'drawings/wave/training',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

wave_test_generator = test_datagen.flow_from_directory(
    'drawings/wave/testing',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    
    for layer in base_model.layers:
        layer.trainable = False

    return model

model = build_model()


model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    verbose=1,
    min_lr=0.00001
)

callbacks_list = [early_stopping, reduce_learning_rate]


steps_per_epoch = max(1, spiral_train_generator.n // spiral_train_generator.batch_size)
validation_steps = max(1, spiral_test_generator.n // spiral_test_generator.batch_size)

history = model.fit(
    spiral_train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=spiral_test_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_list
)

plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.legend(loc='lower right')

plt.show()

# Get the final training and validation accuracy
final_training_accuracy = history.history['accuracy'][-1]
final_validation_accuracy = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {final_training_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_validation_accuracy:.4f}")
