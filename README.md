# CSharpNeuralNetwork

C# collection that provides C# matrix operation and neural network implementation for developing machine learning algorithms and is

- easy to use -> great for small projects or just to learn how machine learning works.
- small and simple -> easy to understand and make changes.
- light weight -> no more than 2000 lines.

## Getting Started

### Prerequisites

This project is written in pure C# without any usage of none standard libraries.

### Installation

Just add all the source code to your project and you can get started. The class "Tester" has basic examples of creating a neural network instance for training and predicting.

## Code Example

### Neural Network

```C#
// Create a new neural network that has 2 input neurons,
// a hidden layer with 10 neurons and an output layer with
// 1 neuron. Both the hidden layer and the output layer
// have no activation function.
NeuralNetwork network = new NeuralNetwork(
    2,
    new[] { 10, 1 },
    new ActivationFunction[]
    {
        new ActivationFunction(ActivationFunction.Type.None),
        new ActivationFunction(ActivationFunction.Type.None)
    });
// Seed the weights with random values.
network.SeedWeights(-1, 1);
Console.WriteLine(network + "\n");

// If you want to use your own KeepTrainingFuncion or ErrorFuncion,
// just set the corresponding delegate property of the instance with
// your own one here. Otherwise it will use the default funcions.

// 10 input sets.
Matrix trainingInput = new Matrix(10, 2);
// 10 output sets.
Matrix trainingOutput = new Matrix(10, 1);

// Create random training sets.
Random random = new Random();
for (int i = 0; i < trainingInput.Height; i++)
{
    trainingInput[i, 0] = random.NextDouble();
    trainingInput[i, 1] = random.NextDouble();

    trainingOutput[i, 0] = (trainingInput[i, 0] + trainingInput[i, 1]) / 2;
}

Console.WriteLine("Training input sets:");
Console.WriteLine(trainingInput + "\n");
Console.WriteLine("Training output sets:");
Console.WriteLine(trainingOutput + "\n");

Console.WriteLine("Cost before training:");
Console.WriteLine(network.Cost(trainingInput, trainingOutput));
Console.WriteLine("Result before training:");
Console.WriteLine(network.Forward(trainingInput) + "\n");

Console.WriteLine("Training...");
// learning Rate: 0.2.
// printToConsole: true.
network.Train(0.2, trainingInput, trainingOutput, true);
Console.WriteLine("Done!" + "\n");

Console.WriteLine("Cost after training:");
Console.WriteLine(network.Cost(trainingInput, trainingOutput));
Console.WriteLine("Result after training:");
Console.WriteLine(network.Forward(trainingInput) + "\n");

// 10 input sets.
Matrix predictionInput = new Matrix(10, 2);
// 10 output sets.
Matrix predictionOutput = new Matrix(10, 1);

// Create random prediction sets.
for (int i = 0; i < predictionInput.Height; i++)
{
    predictionInput[i, 0] = random.NextDouble();
    predictionInput[i, 1] = random.NextDouble();

    predictionOutput[i, 0] = (predictionInput[i, 0] + predictionInput[i, 1]) / 2;
}

Console.WriteLine("Prediction input sets:");
Console.WriteLine(predictionInput + "\n");
Console.WriteLine("Prediction output sets:");
Console.WriteLine(predictionOutput + "\n");

Console.WriteLine("Cost for prediction:");
Console.WriteLine(network.Cost(predictionInput, predictionOutput));
Console.WriteLine("Result for prediction:");
Console.WriteLine(network.Forward(predictionInput) + "\n");
```

## References

This project is inspired by sebig3000's [MachineLearning](https://github.com/sebig3000/MachineLearning). I made a C# version of it and fixed bugs and polished it. Several matrix operation methods are replaced by overridden operators and new operations are added. The way to control the KeepTraining method of class NeuralNetwork is also changed and the way to calculate the error is also adjustable now, using delegate properties to replace the default function.