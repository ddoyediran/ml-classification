import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

// Grab a reference to the MNIST input values (pixel data)
const INPUTS = TRAINING_DATA.inputs;

// Grab a reference to the MNIST output values
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the 2 arrays in the same way so inputs still match outputs indexes
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array is 1 dimensional
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// Output feature Array is 1 dimensional
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

// Create and define model architecture
const model = tf.sequential();

model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

train();

async function train() {
  // Compile the model with the defined optimizer ans specify our loss function to use
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Fit and get the results
  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true, // Ensure data is shuffled again before using each epoch
    validationSplit: 0.2,
    batchSize: 512, // Update weights after every 512 examples.
    epoch: 50, // Go over the data 50 times
    callbacks: { onEpochEnd: logProgress },
  });

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();

  evaluate(); // Once trained we can evaluate the model by testing on some images
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

function evaluate() {
  const OFFSET = Math.floor(Math.random() * INPUTS.length); // Select random  intfrom all example inputs.

  let answer = tf.tidy(function () {
    let newInput = tf.tensor1d(INPUTS[OFFSET]); // convert the random number "OFFSET" to input_tensor

    let output = model.predict(newInput.expandDims()); // evaluate the input
    output.print(); // print the output to the console
    return output.squeeze().argMax();
  });

  answer.array().then(function (index) {
    PREDICTION_ELEMENT.innerText = index;
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index === OUTPUTS[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d");

function drawImage(digit) {
  var imageData = CTX.getImageData(0, 0, 28, 28);

  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255; // Red channel
    imageData.data[i * 4 + 1] = digit[i] * 255; // Green channel
    imageData.data[i * 4 + 2] = digit[i] * 255; // Blue channel
    imageData.data[i * 4 + 3] = 255; // Alpha channel
  }

  // Render the updated array of data to the canvas itself
  CTX.putImageData(imageData, 0, 0);

  // Perform a new classification after a certain interval
  setTimeout(evaluate, 2000);
}

/**
 * To create and test a ML model
 * 1. Import the training data (input and output)
 * 2. Set our Input and output tensor (shuffle the training data if need be)
 * 3. Create the model architecture:
 * a. set the model to be sequential
 * b. Add all the layers (neurons including hidden, output) wit their input shapes?, units "output" and activation function
 * 4. Train the model by:
 * a. Compile the model with optimizer, loss function, metrics
 * b. Fit and get the result from the training which takes 3 parameters: Input_tensor, output_tensor and an object that contains shuffle, validationSplit, batchSize, epoch
 * c. Dispose all the tensors created
 * d. Evaluate the model
 */
