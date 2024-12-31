from tensorflow import keras
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from matplotlib import pyplot as plt
from keras.optimizers import SGD
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

train_data_dir="D:/9raya/DeepLearning/Projet/FaceMaskDataset/train224"
test_data_dir="D:/9raya/DeepLearning/Projet/FaceMaskDataset/test224"

model= keras.models.Sequential()
model.add(keras.layers.Conv2D(64,(3,3),input_shape=(64,64,3),activation="relu",strides=(1,1),padding='valid'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense( 2,activation='softmax'))

# Compiler le modèle
custom_optimizer = SGD(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,      # Normalisation des valeurs de pixel (mise à l'échelle)
    rotation_range=30,      # Rotation aléatoire de l'image de -40 à 40 degrés
    width_shift_range=0.3,  # Déplacement horizontal aléatoire de l'image
    height_shift_range=0.3, # Déplacement vertical aléatoire de l'image
    zoom_range=0.3,         # Zoom aléatoire de l'image
    horizontal_flip=True,   # Retournement horizontal aléatoire de l'image
    fill_mode='nearest'     # Mode de remplissage des pixels après les transformations
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),   # Redimensionnement des images à la taille spécifiée
    batch_size=128,           # Taille du batch d'entraînement
    class_mode='categorical' # Mode de classification catégorielle
)

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(64, 64),
    class_mode='categorical'
)

model_checkpoint = ModelCheckpoint("modelCNN.h5", verbose=1, save_best_only=True, save_weights_only=False,save_freq=1,mode='auto',monitor="accuracy")

# Définir le rappel d'arrêt anticipé
early_stopping = EarlyStopping(monitor="accuracy", min_delta=0, patience=50,verbose=1,mode='auto')

# Entraîner le modèle avec les données d'entraînement et de validation
history = model.fit_generator(
    train_generator,  # Générateur de données d'entraînement
    steps_per_epoch=len(train_generator),  # Nombre d'étapes par époque (nombre total de lots)
    epochs=100,  # Nombre d'époques
    validation_data=validation_generator,  # Générateur de données de validation
    validation_steps=len(validation_generator),
    callbacks=[model_checkpoint,early_stopping]# Nombre d'étapes de validation (nombre total de lots de validation)
)

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')  # Add this line
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

