using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace CSharpNeuralNetwork
{
    /// <summary>
    /// The implementation of a neural network. This class can't be inherited.
    /// </summary>
    public sealed class NeuralNetwork
    {
        /// <summary>
        /// Number of input neurons.
        /// </summary>
        private readonly int _numberOfInputs;

        /// <summary>
        /// Number of inputs.
        /// </summary>
        public int NumberOfInputs => _numberOfInputs;

        /// <summary>
        /// Number of input neurons in each layer.
        /// </summary>
        private readonly int[] _layerSizes;

        /// <summary>
        /// Number of outputs.
        /// </summary>
        public int NumberOfOutputs => _layerSizes[_layerSizes.Length - 1];

        /// <summary>
        /// Number of layers.
        /// </summary>
        public int NumberOfLayers => _layerSizes.Length;

        /// <summary>
        /// Each layer's activation function.
        /// </summary>
        private readonly ActivationFunction[] _activationFunctions;

        /// <summary>
        /// Each layer's activation function.
        /// </summary>
        public ActivationFunction[] ActivationFunctions => _activationFunctions;

        /// <summary>
        /// Weights.
        /// </summary>
        private readonly Matrix[] _weights;

        /// <summary>
        /// Weights.
        /// </summary>
        public Matrix[] Weights => _weights;

        /// <summary>
        /// Activities, needed for backpropagation.
        /// </summary>
        private readonly Matrix[] _activityA, _activityZ;

        /// <summary>
        /// Function that tells the network whether to continue training.
        /// This is set to a default function that returns false if
        /// iterations is no less than 100 or learningRate is no greater
        /// than 0.
        /// </summary>
        private KeepTrainingFunction _keepTrainingFunction;

        /// <summary>
        /// Function that tells the network whether to continue training.
        /// This is set to a default function that returns false if
        /// iterations is no less than 100 or learningRate is no greater
        /// than 0.
        /// </summary>
        public KeepTrainingFunction KeepTrainingFunction => _keepTrainingFunction;

        /// <summary>
        /// Calculates the error of the prediction to the actuality.
        /// Every single row represents one dataset and the two matrix
        /// must be of the same size.
        /// This is set to a default function calculating the mean
        /// square error in the constructor.
        /// </summary>
        private ErrorFuntion _errorFunction;

        /// <summary>
        /// Calculates the error of the prediction to the actuality.
        /// Every single row represents one dataset and the two matrix
        /// must be of the same size.
        /// This is set to a default function calculating the mean
        /// square error in the constructor.
        /// </summary>
        public ErrorFuntion ErrorFunction => _errorFunction;

        /// <summary>
        /// Constructs a new neural network.
        /// </summary>
        /// <param name="numberOfInputs">Number of inputs.</param>
        /// <param name="layerSizes">
        /// Number of neurons in each hidden layer.
        /// The last layer is the output layer.
        /// </param>
        /// <param name="activationFunctions">Activation functions for every layer.</param>
        /// <exception cref="ArgumentException"></exception>
        public NeuralNetwork(int numberOfInputs, int[] layerSizes, ActivationFunction[] activationFunctions)
        {
            if (numberOfInputs < 1)
            {
                throw new ArgumentException("Number of input neurons less than one!");
            }

            if (layerSizes.Length < 1)
            {
                throw new ArgumentException("Number of layers less than one!");
            }

            if (activationFunctions.Length != layerSizes.Length)
            {
                throw new ArgumentException("Number of activation functions doesn't match that of layers");
            }

            foreach (var layerSize in layerSizes)
            {
                if (layerSize < 1)
                {
                    throw new ArgumentException("Number of neurons in layer less than one!");
                }
            }

            _numberOfInputs = numberOfInputs;
            _layerSizes = new int[layerSizes.Length];
            layerSizes.CopyTo(_layerSizes, 0);
            _activationFunctions = new ActivationFunction[activationFunctions.Length];
            for (int i = 0; i < activationFunctions.Length; i++)
            {
                _activationFunctions[i] = new ActivationFunction(activationFunctions[i]);
            }
            _weights = new Matrix[layerSizes.Length];
            _weights[0] = new Matrix(numberOfInputs, layerSizes[0]);
            for (int i = 1; i < layerSizes.Length; i++)
            {
                _weights[i] = new Matrix(layerSizes[i - 1], layerSizes[i]);
            }
            _activityA = new Matrix[layerSizes.Length];
            _activityZ = new Matrix[layerSizes.Length];
            _keepTrainingFunction = DefaultKeepTraining;
            _errorFunction = DefaultError;
        }

        /// <summary>
        /// Constructs a new neural network based on a given one.
        /// </summary>
        /// <param name="input">The given neural network.</param>
        public NeuralNetwork(NeuralNetwork input)
        {
            _numberOfInputs = input._numberOfInputs;
            _layerSizes = new int[input._layerSizes.Length];
            input._layerSizes.CopyTo(_layerSizes, 0);
            _activationFunctions = new ActivationFunction[input._activationFunctions.Length];
            for (int i = 0; i < input._activationFunctions.Length; i++)
            {
                _activationFunctions[i] = new ActivationFunction(input._activationFunctions[i]);
            }
            _weights = new Matrix[input._layerSizes.Length];
            for (int i = 0; i < input._activationFunctions.Length; i++)
            {
                _weights[i] = new Matrix(input._weights[i]);
            }
            _activityA = new Matrix[input._layerSizes.Length];
            _activityZ = new Matrix[input._layerSizes.Length];
            _keepTrainingFunction = input._keepTrainingFunction;
            _errorFunction = input._errorFunction;
        }

        /// <summary>
        /// Returns the number of neurons of the specified layer.
        /// </summary>
        /// <param name="index">Index of the layer.</param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <returns>Number of neurons in the layer.</returns>
        public int GetLayerSize(int index)
        {
            if (index < 0 || index >= _activationFunctions.Length)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            return _layerSizes[index];
        }

        /// <summary>
        /// Returns a copy of the numbers of neurons in every layer.
        /// </summary>
        /// <returns>Copy of numbers of neurons in every layer.</returns>
        public int[] CopyLayerSizes()
        {
            int[] copy = new int[_layerSizes.Length];
            _layerSizes.CopyTo(copy, 0);
            return copy;
        }

        /// <summary>
        /// Set the activation function of the specified layer.
        /// </summary>
        /// <param name="index">Index of the layer.</param>
        /// <param name="activationFunction">Activation function.</param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public void SetActivationFunction(int index, ActivationFunction activationFunction)
        {
            if (index < 0 || index >= _activationFunctions.Length)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            _activationFunctions[index].ChangeType(activationFunction.CurrentType);
        }

        /// <summary>
        /// Returns the activation function of the specified layer.
        /// </summary>
        /// <param name="index">Index of the layer.</param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <returns>Activation function of the layer.</returns>
        public ActivationFunction GetActivationFunction(int index)
        {
            if (index < 0 || index >= _activationFunctions.Length)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            return _activationFunctions[index];
        }

        /// <summary>
        /// Returns a copy of the activation functions of every layer.
        /// </summary>
        /// <returns></returns>
        public ActivationFunction[] CopyActivationFunctions()
        {
            ActivationFunction[] result = new ActivationFunction[_activationFunctions.Length];

            for (int i = 0; i < _activationFunctions.Length; i++)
            {
                result[i] = new ActivationFunction(_activationFunctions[i]);
            }

            return result;
        }

        /// <summary>
        /// Sets the weights of a single layer.
        /// </summary>
        /// <param name="index">Index of the layer.</param>
        /// <param name="layer">New weights.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public void SetWeights(int index, Matrix layer)
        {
            if (index < 0 || index >= _weights.Length)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            if (layer.Height != _weights[index].Height || layer.Width != _weights[index].Width)
            {
                throw new ArgumentException("Incorrect layer dimensions!");
            }

            _weights[index] = new Matrix(layer);
        }

        /// <summary>
        /// Sets weights of every layer.
        /// </summary>
        /// <param name="weights">New weights.</param>
        /// <exception cref="ArgumentException"></exception>
        public void SetWeights(Matrix[] weights)
        {
            if (weights.Length != _weights.Length)
            {
                throw new ArgumentException("Incorrect number of layers!");
            }

            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i].Height != _weights[i].Height || weights[i].Width != _weights[i].Width)
                {
                    throw new ArgumentException("Incorrect layer dimensions!");
                }
            }

            for (int i = 0; i < weights.Length; i++)
            {
                _weights[i] = new Matrix(weights[i]);
            }
        }

        /// <summary>
        /// Returns the weight matrix of a specified layer.
        /// </summary>
        /// <param name="index">Index of the layer.</param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <returns>Weights of the layer.</returns>
        public Matrix GetWeights(int index)
        {
            if (index < 0 || index >= _weights.Length)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            return _weights[index];
        }

        /// <summary>
        /// Returns a copy of all weights.
        /// </summary>
        /// <returns>Copy of all weights.</returns>
        public Matrix[] CopyWeights()
        {
            Matrix[] copy = new Matrix[_weights.Length];

            for (int i = 0; i < _weights.Length; i++)
            {
                copy[i] = new Matrix(_weights[i]);
            }

            return copy;
        }

        /// <summary>
        /// Seed the weights within the given boundaries.
        /// </summary>
        /// <param name="minimum">Minimum value.</param>
        /// <param name="maximum">Maximum value.</param>
        public void SeedWeights(double minimum, double maximum)
        {
            Random random = new Random();

            foreach (var layer in _weights)
            {
                layer.FillRand(random, minimum, maximum);
            }
        }

        /// <summary>
        /// Seeds the weights based on a seed and between the given boundaries.
        /// </summary>
        /// <param name="seed">Seed for the random number generator.</param>
        /// <param name="minimum">Minimum value.</param>
        /// <param name="maximum">Maximum value.</param>
        public void SeedWeights(int seed, double minimum, double maximum)
        {
            Random random = new Random(seed);

            foreach (var layer in _weights)
            {
                layer.FillRand(random, minimum, maximum);
            }
        }

        /// <summary>
        /// Seeds the weights within the given boundaries for each layer.
        /// </summary>
        /// <param name="minimums">Minimum values for each layer.</param>
        /// <param name="maximums">Maximum values for each layer.</param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public void SeedWeights(double[] minimums, double[] maximums)
        {
            if (minimums.Length != _weights.Length || maximums.Length != _weights.Length)
            {
                throw new IndexOutOfRangeException("Illegal number of boundaries!");
            }

            Random random = new Random();

            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i].FillRand(random, minimums[i], maximums[i]);
            }
        }

        /// <summary>
        /// Seeds the weights based on a seed and within the given boundaries for each layer.
        /// </summary>
        /// <param name="seed">The given seed.</param>
        /// <param name="minimums">Minimum values for each layer.</param>
        /// <param name="maximums">Maximum values for each layer.</param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public void SeedWeights(int seed, double[] minimums, double[] maximums)
        {
            if (minimums.Length != _weights.Length || maximums.Length != _weights.Length)
            {
                throw new IndexOutOfRangeException("Illegal number of boundaries!");
            }

            Random random = new Random(seed);

            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i].FillRand(random, minimums[i], maximums[i]);
            }
        }

        /// <summary>
        /// Eliminates infinite numbers and Nans.
        /// </summary>
        public void KeepWeightsInBounds()
        {
            foreach (var layer in _weights)
            {
                layer.Apply(x =>
                {
                    if (double.IsNaN(x))
                    {
                        return 0.0;
                    }
                    else if (x <= double.NegativeInfinity)
                    {
                        return double.MinValue;
                    }
                    else if (x >= double.PositiveInfinity)
                    {
                        return double.MaxValue;
                    }
                    else
                    {
                        return x;
                    }
                });
            }
        }

        /// <summary>
        /// Eliminates infinite numbers and Nans.
        /// </summary>
        /// <param name="minimum">Minimum value.</param>
        /// <param name="maximum">Maximum value.</param>
        /// <exception cref="ArgumentException"></exception>
        public void KeepWeightsInBounds(double minimum, double maximum)
        {
            if (minimum >= maximum)
            {
                throw new ArgumentException("Minimum greater than or equal to maximum!");
            }

            foreach (var layer in _weights)
            {
                layer.Apply(x =>
                {
                    if (double.IsNaN(x))
                    {
                        return minimum / 2 + maximum / 2;
                    }
                    else if (x < minimum)
                    {
                        return minimum;
                    }
                    else if (x > maximum)
                    {
                        return maximum;
                    }
                    else
                    {
                        return x;
                    }
                });
            }
        }

        /// <summary>
        /// Forward propagates a matrix of datasets.
        /// Every single row represents one dataset.
        /// Every column gets feed into one input neuron.
        /// </summary>
        /// <param name="input">Input datasets</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>Output sets.</returns>
        public Matrix Forward(Matrix input)
        {
            if (input.Width != _numberOfInputs)
            {
                throw new ArgumentException();
            }

            _activityZ[0] = input.Dot(_weights[0]);
            _activityA[0] = _activityZ[0].Apply(_activationFunctions[0].Function);

            for (int i = 1; i < _weights.Length; i++)
            {
                _activityZ[i] = _activityA[i - 1].Dot(_weights[i]);
                _activityA[i] = _activityZ[i].Apply(_activationFunctions[i].Function);
            }

            return new Matrix(_activityA[_weights.Length - 1]);
        }

        /// <summary>
        /// Calculates the mean square error of the prediction to the actuality.
        /// Every single row represents one dataset and the two matrix must be
        /// of the same size.
        /// </summary>
        /// <param name="prediction">Prediction sets</param>
        /// <param name="actuality">Actuality sets</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The mean square error.</returns>
        public static double DefaultError(Matrix prediction, Matrix actuality)
        {
            if (prediction.Height != actuality.Height || prediction.Width != actuality.Width)
            {
                throw new ArgumentException("Dimensions not compatible!");
            }

            Matrix difference = actuality - prediction;
            Matrix squareError = difference * difference;

            double error = 0;
            for (int i = 0; i < squareError.Height; i++)
            {
                for (int j = 0; j < squareError.Width; j++)
                {
                    error += squareError[i, j];
                }
            }

            error /= 2;
            error /= actuality.Height;

            return error;
        }

        /// <summary>
        /// Calculates the error of the prediction to the given output
        /// using "ErrorFunction".
        /// Every single row represents one dataset.
        /// Every column gets feed into one input/output neuron.
        /// </summary>
        /// <param name="input">Input sets.</param>
        /// <param name="output">Output sets.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>Error.</returns>
        public double Cost(Matrix input, Matrix output)
        {
            if (input.Width != NumberOfInputs)
            {
                throw new ArgumentException("Illegal number of inputs!");
            }   

            if (output.Width != NumberOfOutputs)
            {
                throw new ArgumentException("Illegal number of outputs!");
            }

            if (input.Height != output.Height)
            {
                throw new ArgumentException("Unequal number of input anf output sets!");
            }

            Matrix yHat = Forward(input);

            return DefaultError(yHat, output);
        }

        /// <summary>
        /// Backpropagates the error to every weight.
        /// Every single row represents one dataset.
        /// Every column gets feed into one input/output neuron.
        /// </summary>
        /// <param name="input">Input sets.</param>
        /// <param name="output">Output sets.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>Derivative of the error to every weight.</returns>
        public Matrix[] CostPrime(Matrix input, Matrix output)
        {
            if (input.Width != NumberOfInputs)
            {
                throw new ArgumentException("Illegal number of inputs!");
            }

            if (output.Width != NumberOfOutputs)
            {
                throw new ArgumentException("Illegal number of outputs!");
            }

            if (input.Height != output.Height)
            {
                throw new ArgumentException("Unequal number of input anf output sets!");
            }

            Matrix[] dJdW = new Matrix[_weights.Length];
            Matrix yHat = Forward(input);
            Matrix delta = (yHat - output) *
                           _activityZ[_weights.Length - 1].Apply(_activationFunctions[_weights.Length - 1].Prime);

            for (int i = _weights.Length - 1; i > 0; i--)
            {
                dJdW[i] = _activityA[i - 1].Transpose().Dot(delta);
                delta = delta.Dot(_weights[i].Transpose()) * _activityZ[i - 1].Apply(_activationFunctions[i - 1].Prime);
            }
            dJdW[0] = input.Transpose().Dot(delta);
            return dJdW;
        }

        /// <summary>
        /// Backpropagates and applies the gradient with the given learning rate once.
        /// </summary>
        /// <param name="learningRate">The given learning rate.</param>
        /// <param name="input">Input sets</param>
        /// <param name="output">Output sets.</param>
        private void SingleGradientDescent(double learningRate, Matrix input, Matrix output)
        {
            Matrix[] dJdW = CostPrime(input, output);

            for (int i = 0; i < _weights.Length; i++)
            {
                Matrix update = dJdW[i] * (-learningRate);
                _weights[i] += update;
            }

            KeepWeightsInBounds();
        }

        /// <summary>
        /// Tells the network whether to continue training in the default condition.
        /// </summary>
        /// <param name="iterations">Number of completed training cycles.</param>
        /// <param name="learningRate">Current learning rate.</param>
        /// <param name="cost">Current cost.</param>
        /// <returns>Whether the training process should continue.</returns>
        public static bool DefaultKeepTraining(int iterations, double learningRate, double cost)
        {
            return iterations < 100 && learningRate > 0;
        }

        /// <summary>
        /// Trains the network.
        /// </summary>
        /// <param name="learningRate">Initial learning rate.</param>
        /// <param name="input">Input sets.</param>
        /// <param name="output">Expected output sets.</param>
        /// <param name="printToConsole">Print progress to console.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>Last cost.</returns>
        public double Train(double learningRate, Matrix input, Matrix output, bool printToConsole)
        {
            if (input.Width != NumberOfInputs)
            {
                throw new ArgumentException("Illegal number of inputs!");
            }

            if (output.Width != NumberOfOutputs)
            {
                throw new ArgumentException("Illegal number of outputs!");
            }

            if (input.Height != output.Height)
            {
                throw new ArgumentException("Unequal number of input anf output sets!");
            }

            double lastCost = Cost(input, output);

            for (int iterations = 0; _keepTrainingFunction(iterations, learningRate, lastCost); iterations++)
            {
                Matrix[] lastWeights = CopyWeights();

                SingleGradientDescent(learningRate, input, output);
                double currentCost = Cost(input, output);

                if (printToConsole)
                {
                    Console.WriteLine("{0:D}: {1:G}", iterations, currentCost);
                }

                if (currentCost <= lastCost)
                {
                    lastCost = currentCost;
                    learningRate *= 1.1;
                }
                else
                {
                    SetWeights(lastWeights);
                    learningRate /= 2;
                }
            }

            return lastCost;
        }

        /// <summary>
        /// Returns the string form of this neural network.
        /// </summary>
        /// <returns>The string form of this neural network</returns>
        public override string ToString()
        {
            StringBuilder result = new StringBuilder("Neural network {");
            result.Append(NumberOfInputs);
            result.Append(" ");
            result.Append(_layerSizes);
            result.Append("\n");

            for (int i = 0; i < NumberOfLayers; i++)
            {
                result.Append(_activationFunctions[i]);
                result.Append("\n");
                result.Append(_weights[i]);
                result.Append("\n");
            }

            result.Append("}");

            return result.ToString();
        }
    }
}
