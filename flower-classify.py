 #imports
import time
from time import sleep
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import json
from PIL import Image


print()
print()
print()
print()

print('Welcome to the Flower Classifer! Please follow the steps to classify your flower.')
time.sleep(1)

#select image
print()
print('1. Enter the path of prediction image')
image_path = input()
print()

#select model
def model_selection():
    print('2. Which neural network would you like to use? (Enter 4, 6, or 8) The number corresponds to the number of epochs the network was trained on.')
    model = input()
    while model != '4' and model != '6' and model != '8':
        print('Invalid model. Try again')
        model = input()
    if model == '4':
        model = 'neural-networks/four_epoch.h5'
        return model
    elif model == '6':
        model = 'neural-networks/six_epoch.h5'
        return model
    elif model == '8':
        model = 'neural-networks/eight_epoch.h5'
        return model

model = model_selection()
print()

#select prediciton count
print('3. Enter the desired number of top predictions')
top_k = input()
top_k = int(top_k)
print()

print('CALCULATING {} PROBABILITIES FOR {} using {} NEURAL NETWORK'.format(top_k, image_path, model))
sleep(3)

#load json dictionary
category_names = 'label_map.json'
with open(category_names, 'r') as f:
    class_names = json.load(f)

#load model
def load_model(model):
    loaded = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    return loaded
loaded_model = load_model(model)

#process image function
def process_image(input_array):
    tensor = tf.convert_to_tensor(input_array)
    resized = tf.image.resize(tensor, (224, 224))
    normalized = tf.math.divide(resized, 255)
    final = normalized.numpy()
    return final

#predict function
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    with_1 = np.expand_dims(processed_test_image, axis=0)
    to_tensor = tf.convert_to_tensor(with_1)

    predictions = loaded_model.predict(to_tensor)[0]
    labels_unfinished = (-predictions).argsort()[:top_k]
    labels_complete = [x + 1 for x in labels_unfinished]

    preds = []
    for i in labels_unfinished:
        preds.append(predictions[i])


    return preds, labels_complete

#output
probs, classes = predict(image_path, model, top_k)
classes_with_names = []
for i in range(len(classes)):
    classes_with_names.append(class_names[str(classes[i])])

print()
print('~~~RESULTS~~~')
for i in range(len(probs)):
    percentage = round((probs[i] * 100), 2)
    species = classes_with_names[i]
    print('{}% percent chance of {}'.format(percentage, species))
    print()
    time.sleep(1)
