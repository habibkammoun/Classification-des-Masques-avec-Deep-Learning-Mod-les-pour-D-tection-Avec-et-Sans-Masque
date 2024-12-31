import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
from keras_preprocessing.image import load_img, img_to_array

model = load_model("lahbib.h5")

def predict(model, img_path):
    img = load_img(img_path, target_size=(64, 64))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

# Example usage
image_path_to_classify = "azzasansbavette.jpeg"
prediction = predict(model, image_path_to_classify)

# The prediction contains the probability for each class
# For binary classification, you can check the probability for class 1
if prediction[0] >= 0.5:
    print("Class: Face")
else:
    print("Class: Maskface")
