## Title

Multi-layer perceptrons for classification - Implementing a classifier in TensorFlow.js

## About the project

Implementation of a multi layer perceptron that can classify digits (hand written number) of the famous MNIST dataset right in your web browser.

Key things: transforming images to Tensors, and understanding the new outputs of the model.

### To create and test a ML model

1.  Import the training data (input and output)
2.  Set our Input and output tensor (shuffle the training data if need be)
3.  Create the model architecture:

- set the model to be sequential
- Add all the layers (neurons including hidden, output) wit their input shapes?, units "output" and activation function

4.  Train the model by:

- Compile the model with optimizer, loss function, metrics
- Fit and get the result from the training which takes 3 parameters: Input_tensor, output_tensor and an object that contains shuffle, validationSplit, batchSize, epoch
- Dispose all the tensors created
- Evaluate the model
