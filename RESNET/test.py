import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
from keras_preprocessing.image import load_img
from keras_preprocessing import image
import os

model = load_model("lahbib.h5")

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

text_file1 = open("falsedetectionCNNface.txt", "w")
img_path = "D:/9raya/DeepLearning/Projet/FaceMaskDataset/test224/face"

vrai = 0
total = 0
for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff')):
        continue
    print(img_name)
    filepath = os.path.join(img_path, img_name)
    img = load_img(filepath, target_size=(224, 224))  # Update target_size
    total += 1
    preds = predict(model, img)
    if preds[0] >= 0.5:
        vrai += 1
    else:
        text_file1.write(filepath + "\n")
    print(preds)

acc1 = (vrai / total) * 100
text_file1.close()

text_file2 = open("falsedetectionCNNmaskface.txt", "w")
img_path = "C:/Users/habou/PycharmProjects/TP5/FaceMaskDataset/test224/maskface"

vrai = 0
total = 0
for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff')):
        continue
    print(img_name)
    filepath = os.path.join(img_path, img_name)
    img = load_img(filepath, target_size=(224, 224))  # Update target_size
    total += 1
    preds = predict(model, img)
    if preds[1] >= 0.5:
        vrai += 1
    else:
        text_file2.write(filepath + "\n")
    print(preds)

acc2 = (vrai / total) * 100
print("acc face : " + str(acc1))
print("acc maskface : " + str(acc2))
text_file2.close()
