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
            NeuralNetwork network = new NeuralNetwork(
                2,
                new[] { 10, 1 },
                new ActivationFunction[]
                {
                    new ActivationFunction(ActivationFunction.Type.None),
                    new ActivationFunction(ActivationFunction.Type.None)
                });
            network.SeedWeights(-1, 1);
            Console.WriteLine(network + "\n");      

            Matrix trainingInput = new Matrix(10, 2);
            Matrix trainingOutput = new Matrix(10, 1);

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
            network.Train(0.2, trainingInput, trainingOutput, true);
            Console.WriteLine("Done!" + "\n");

            Console.WriteLine("Cost after training:");
            Console.WriteLine(network.Cost(trainingInput, trainingOutput));
            Console.WriteLine("Result after training:");
            Console.WriteLine(network.Forward(trainingInput) + "\n");

            Matrix predictionInput = new Matrix(10, 2);
            Matrix predictionOutput = new Matrix(10, 1);

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
