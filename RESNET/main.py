from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
train_data_dir = 'D:/9raya/DeepLearning/Projet/FaceMaskDataset/train224'
validation_data_dir = 'D:/9raya/DeepLearning/Projet/FaceMaskDataset/test224'
batch_size = 4
epochs = 50
img_width, img_height = 224, 224

# Load the ResNet50 model pre-trained on ImageNet data
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the Layers which you don't want to train. Here I am freezing all layers except the last ones.
base_model.layers[0].trainable = False

# Create a custom model
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(inputs=base_model.input, outputs=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rescaling for validation set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

checkpoint = ModelCheckpoint("lahbib.h5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
early = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

history = model_final.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early]
)

# Print a summary of the ResNet50 model architecture
model_final.summary()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()