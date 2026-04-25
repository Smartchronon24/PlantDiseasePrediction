# LeafDoctor: Deep Learning for Crop Disease Classification

![Plant Disease Detection Banner](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-orange.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)

**LeafDoctor** is a Deep Learning Image Classification pipeline designed to identify various plant diseases from images of plant leaves. It leverages a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images into **38 different classes** (including various diseases and healthy leaves).

## 🚀 Features

- **Deep CNN Architecture:** A sequential CNN model with 5 convolutional blocks, max-pooling, and dropout layers to prevent overfitting.
- **Robust Image Processing:** Automatically resizes input images to 128x128 pixels and loads them in batches.
- **Comprehensive Evaluation:** Generates precision/recall reports, training accuracy plots, and detailed Confusion Matrices using Scikit-Learn and Seaborn.
- **Ready-to-use Inference:** Contains interactive notebooks and scripts to load the saved model and predict diseases from unseen images.

## 📁 Project Structure

- `PlantMain.py`: A consolidated Python script containing the entire pipeline—from loading data, building the model, training, and evaluating, to testing single images.
- `Train_plant_disease.ipynb`: An interactive Jupyter Notebook for exploring data, training the CNN, and visualizing plots inline.
- `TestModel.ipynb`: A separate notebook focused on loading the saved `model.keras` and running inferences on test images.
- `requirements.txt`: Lists all the necessary dependencies to run the project.

## 🛠️ Tech Stack

- **Machine Learning / Deep Learning:** TensorFlow, Keras, Scikit-learn
- **Image Processing:** OpenCV (`cv2`)
- **Data Manipulation & Visualization:** NumPy, Pandas, Matplotlib, Seaborn

## 🧠 Model Architecture

The neural network is built with the following structure:
1. **Feature Extraction:** 5 Convolutional Blocks. Each block has two `Conv2D` layers followed by a `MaxPool2D` layer. Filter sizes increase progressively (32 → 64 → 128 → 256 → 512).
2. **Regularization:** Includes `Dropout(0.25)` and `Dropout(0.4)` to combat overfitting.
3. **Classification:** A flattened vector passed through a dense layer with 1500 units.
4. **Output Layer:** A final dense layer with 38 units and a `softmax` activation function, corresponding to the 38 possible plant disease classifications.

The model is compiled using the **Adam optimizer** (`learning_rate=0.0001`) and the `categorical_crossentropy` loss function.

## ⚙️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Smartchronon24/PlantDiseasePrediction.git
   cd PlantDiseasePrediction
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Setup:**
   Ensure you have the dataset placed in a `dataset/` directory with the following structure:
   ```
   dataset/
   ├── train/
   ├── valid/
   └── test/
   ```

4. **Run the Model:**
   You can either run the complete pipeline using the Python script or interactively via Jupyter Notebooks:
   ```bash
   python PlantMain.py
   ```

## 📊 Results and Visualization

During training, the script generates a `training_history.json` file which is used to plot accuracy over time. It also generates a large heatmap of the **Confusion Matrix** to help visualize which plant diseases the model might be confusing with one another.

---
*Created as part of an exploration into computer vision and agricultural tech.*
