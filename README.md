# Fruit Image Classification using Convolutional Neural Networks (CNN)
This repository provides a Convolutional Neural Network (CNN) model trained to classify fruit and vegetable images. The model classifies photos of fruits into groups based on their names.

## Dataset
The <a href='https://www.kaggle.com/datasets/moltean/fruits'>Fruit 360</a> dataset is used to train and evaluate the model. The dataset contains approximately 90,400+ photos of different fruits and 
vegetables. The collection contains 131 distinct fruit classifications. The categories are separated into folders, each containing photos from a certain fruit group. Both the training and testing sets 
have 131 fruit classes.

## Model Architecture.
This project's CNN model architecture is implemented with TensorFlow's tf.keras.Sequential() API. The design starts with convolutional (Conv2D) layers, which extract features from input pictures. 
Pooling layers (MaxPooling2D) are used to flatten and downsample features, preserving crucial information while reducing spatial dimensions. After flattening feature maps into a 1D vector, 
fully connected (Dense) layers are used to classify retrieved features. The output layer produces the final predictions, using a softmax activation function for multi-class classification.

The model architecture is intended to learn attributes from input photos and categorize them into several fruit types.

## Training and Evaluation
### Training
The dataset was split into training and validation sets using common practice, with a larger portion allocated for training to ensure the model learns from a diverse 
range of examples. During training, the model iteratively processed batches of images, adjusting its internal parameters (weights and biases) using backpropagation and Adam optimizer for
optimization. During the training process, I aimed to minimize loss function using categorical cross-entropy (for multi-class classification tasks), while monitoring performance on 
the validation set to detect overfitting.

### Evaluation
Evaluation criteria including as accuracy, precision, recall, and F1-score are used to assess the model's performance across classes. I avoided using a confusion 
matrix due to the dataset's large number of classes. These metrics provide insights into the model's ability to correctly classify images into their respective categories. 



