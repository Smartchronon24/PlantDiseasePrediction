
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2

# Training image pre-processing
training_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


model = Sequential()

# Building convolution Layer
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128,128,3]))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=1500, activation='relu'))
model.add(Dropout(0.4))

#output layer
model.add(Dense(units=38, activation='softmax'))

# Compiling Model
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Model evaluation on training set
training_loss, training_accuracy = model.evaluate(training_set)

print(training_loss, training_accuracy)

# Model evaluation on validation set
training_loss, training_accuracy = model.evaluate(validation_set)

print(training_loss, training_accuracy)

model.save("model.keras")

training_history.history

#Recording history
import json
with open("training_history.json", "w") as f:
    json.dump(training_history.history, f)

epochs = [i for i in range(1, 11)]
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation accuracy')
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy percentage")
plt.title("Visualization of Accuracy")
plt.legend()
plt.show()

class_name = validation_set.class_names

test_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


prediction = model.predict(test_set)

predicted_categories = tf.argmax(prediction, axis=1)

true_categories = tf.concat([y for x,y in test_set], axis=0)

true_categories
y_true = tf.argmax(true_categories, axis=1)

print(classification_report(y_true, predicted_categories, target_names=class_name))
cn = confusion_matrix(y_true, predicted_categories)
cn

# Visualization
plt.figure(figsize=(40,40))
sns.heatmap(cn, annot=True, annot_kws={'size':10})
plt.xlabel("Predicted Class", fontsize=40)
plt.ylabel("Real Class", fontsize=40)
plt.title("Confusion Martrix", fontsize=50)
plt.show()







# ==============================
#       TESTING THE MODEL
# ==============================


model = tf.keras.models.load_model("model.keras")


image_path = "dataset/test/test/AppleCedarRust4.JPG"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()

image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128,128))
input_array = tf.keras.preprocessing.image.img_to_array(image)
input_array = np.array([input_array])
print(input_array.shape)

prediction = model.predict(input_array)
result_index = np.argmax(prediction)
result_index

validation_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)
class_name = validation_set.class_names

model_prediction = class_name[result_index]
print(result_index, model_prediction)


