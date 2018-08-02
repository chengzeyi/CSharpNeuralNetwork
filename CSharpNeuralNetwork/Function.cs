using System;
using System.Collections.Generic;
using System.Text;

namespace CSharpNeuralNetwork
{
    /// <summary>
    /// Function that receives and returns a double.
    /// </summary>
    /// <param name="param">The only parameter.</param>
    /// <returns>The return value.</returns>
    public delegate double DoubleFunction(double param);

    /// <summary>
    /// Function that receives two doubles and returns a double.
    /// </summary>
    /// <param name="paramA">Parameter A.</param>
    /// <param name="paramB">Parameter B.</param>
    /// <returns>The return value.</returns>
    public delegate double BiDoubleFunction(double paramA, double paramB);

    /// <summary>
    /// Function that tells the network whether to continue training.
    /// </summary>
    /// <param name="iterations">Number of completed training cycles.</param>
    /// <param name="learningRate">Current learning rate.</param>
    /// <param name="cost">Current cost.</param>
    /// <returns>Whether the training process should continue.</returns>
    public delegate bool KeepTrainingFunction(int iterations, double learningRate, double cost);

    /// <summary>
    /// Calculates the error of the prediction to the actuality.
    /// Every single row represents one dataset and the two matrix
    /// must be of the same size.
    /// </summary>  
    /// <param name="prediction">Prediction sets.</param>
    /// <param name="actuality">Actuality sets.</param>
    /// <returns>Error.</returns>
    public delegate double ErrorFuntion(Matrix prediction, Matrix actuality);
}
