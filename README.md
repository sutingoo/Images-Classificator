# Introduction
This repository is what I learned from the youtuber RingaTech about Images Classification. I learned from his code and replicate but modified some little things.


# Image Classification with TensorFlow - Fashion MNIST

This project implements a neural network using TensorFlow to classify images from the **Fashion MNIST** dataset. The model is trained, evaluated, and its predictions are visualized through annotated images and probability bar charts.

---

# Dataset

Fashion MNIST is a dataset of 70,000 grayscale images of clothing items (28x28 pixels), split into:

- 60,000 training images
- 10,000 test images

Each image belongs to one of the following 10 classes:

| Label | Class        |
|------:|--------------|
| 0     | T-shirt/top  |
| 1     | Trouser      |
| 2     | Pullover     |
| 3     | Dress        |
| 4     | Coat         |
| 5     | Sandal       |
| 6     | Shirt        |
| 7     | Sneaker      |
| 8     | Bag          |
| 9     | Ankle boot   |

---

# What This Project Does

- Loads and normalizes the Fashion MNIST dataset
- Displays example images from the training set
- Builds and trains a neural network using TensorFlow
- Plots training loss over epochs
- Makes predictions on a batch of test images
- Visualizes predictions with:
  - Image and label comparison (correct vs incorrect)
  - Bar charts showing prediction confidence


# Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
