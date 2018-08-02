using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace CSharpNeuralNetwork
{
    /// <summary>
    /// Several test methods. This class can't be inherited.
    /// </summary>
    public sealed class Tester
    {
        /// <summary>
        /// Tests matrix.
        /// </summary>
        public static void TestMatrix()
        {
            double scalar = 2;
            Matrix matrixA = new Matrix(new double[,]
            {
                {1, 2, 3 },
                {4, 42, 6 },
                {7, 8, 9 }
            });
            Matrix matrixB = new Matrix(new double[,]
            {
                {1, 4, 7 },
                {2, 5, 8 },
                {3, 6, 9 }
            });

            Console.WriteLine("Scalar:");
            Console.WriteLine(scalar + "\n");
            Console.WriteLine("Matrix A:");
            Console.WriteLine(matrixA + "\n");
            Console.WriteLine("Matrix B:");
            Console.WriteLine(matrixB + "\n");
            Console.WriteLine();

            Console.WriteLine("Set [1, 1] to 5:");
            matrixA[1, 1] = 5;
            Console.WriteLine(matrixA + "\n");
            Console.WriteLine("Get [1, 1]: " + matrixA[1, 1]);
            Console.WriteLine("Height: " + matrixA.Height);
            Console.WriteLine("Width: " + matrixA.Width + "\n\n");

            Console.WriteLine("Addition (Matrix A & Matrix B):");
            Console.WriteLine(matrixA + matrixB + "\n");
            Console.WriteLine("Subtraction (Matrix A & Matrix B):");
            Console.WriteLine(matrixA - matrixB + "\n");
            Console.WriteLine("Elementwise multiplication:");
            Console.WriteLine(matrixA * matrixB + "\n");
            Console.WriteLine("Elementwise division:");
            Console.WriteLine(matrixA / matrixB + "\n");
            Console.WriteLine("Matrix multiplication:");
            Console.WriteLine(matrixA.Dot(matrixB) + "\n");
            Console.WriteLine("Scalar multiplication:");
            Console.WriteLine(matrixA * scalar + "\n");
            Console.WriteLine("Scalar division:");
            Console.WriteLine(matrixA / scalar + "\n");

            Console.WriteLine("Applying sine:");
            Console.WriteLine(matrixA.Apply(x => Math.Sin(x)) + "\n\n");

            Console.WriteLine("Transpose:");
            Console.WriteLine(matrixA.Transpose() + "\n\n");

            Console.WriteLine("Get row 1:");
            Console.WriteLine(matrixA.GetRow(1) + "\n");
            Console.WriteLine("Get row 1 & 2:");
            Console.WriteLine(matrixA.GetRows(1, 2));
            Console.WriteLine("Append rows:");
            Console.WriteLine(matrixA.AppendRows(matrixB) + "\n");
            Console.WriteLine("Remove row 1:");
            Console.WriteLine(matrixA.RemoveRow(1) + "\n");
            Console.WriteLine("Remove row 0 & 1:");
            Console.WriteLine(matrixA.RemoveRows(0, 2));

            Console.WriteLine("Get column 1:");
            Console.WriteLine(matrixA.GetColumn(1) + "\n");
            Console.WriteLine("Get column 1 & 2:");
            Console.WriteLine(matrixA.GetColumns(1, 2));
            Console.WriteLine("Append columns:");
            Console.WriteLine(matrixA.AppendColumns(matrixB) + "\n");
            Console.WriteLine("Remove column 1:");
            Console.WriteLine(matrixA.RemoveColumn(1) + "\n");
            Console.WriteLine("Remove column 0 & 1:");
            Console.WriteLine(matrixA.RemoveColumns(0, 2));

            Console.WriteLine("Randomze with [-1, 1]:");
            matrixA.FillRand(new Random(), -1, 1);
            Console.WriteLine(matrixA + "\n");
        }

        /// <summary>
        /// Tests neural network.
        /// </summary>
        public static void TestNeuralNetwork()
        {
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

            // 20 input sets.
            Matrix trainingInput = new Matrix(20, 2);
            // 20 output sets.
            Matrix trainingOutput = new Matrix(20, 1);

            // Create random training sets.
            Random random = new Random();
            for (int i = 0; i < trainingInput.Height; i++)
            {
                trainingInput[i, 0] = random.NextDouble();
                trainingInput[i, 1] = random.NextDouble();

                trainingOutput[i, 0] = Math.Sin(trainingInput[i, 0] + trainingInput[i, 1]);
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

                predictionOutput[i, 0] = Math.Sin(predictionInput[i, 0] + predictionInput[i, 1]);
            }

            Console.WriteLine("Prediction input sets:");
            Console.WriteLine(predictionInput + "\n");
            Console.WriteLine("Prediction output sets:");
            Console.WriteLine(predictionOutput + "\n");

            Console.WriteLine("Cost for prediction:");
            Console.WriteLine(network.Cost(predictionInput, predictionOutput));
            Console.WriteLine("Result for prediction:");
            Console.WriteLine(network.Forward(predictionInput) + "\n");
        }
    }
}
