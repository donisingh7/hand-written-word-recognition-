# hand-written-word-recognition-

This code is for a handwritten digits recognition project using convolutional neural networks (CNNs) in TensorFlow and Keras libraries. The goal of this project is to train a CNN model to recognize handwritten digits from the MNIST dataset and then use the trained model to predict the handwritten digit from an input image.

The code first imports the necessary libraries and defines the CNN model. The model has four layers: two convolutional layers with ReLU activation function and max pooling layers, a flatten layer, and two dense layers with ReLU and softmax activation functions, respectively. The model is then compiled with the Adam optimizer and categorical cross-entropy loss function.

Next, the code loads and preprocesses the MNIST dataset using ImageDataGenerator from Keras. The dataset is split into training and testing sets with a batch size of 32 and a target size of 28x28.

The trained model is then fitted to the training data for 20 epochs, and the validation set is used to evaluate the performance of the model.

Finally, an input image is provided to the model for prediction, and the predicted digit is printed along with the probability distribution of all the digits.
