using System;
using System.Collections.Generic;
using System.Text;

namespace CSharpNeuralNetwork
{
    /// <summary>
    /// Contains name, function, prime of the activation function.
    /// This class can't be inherited.
    /// </summary>
    public sealed class ActivationFunction
    {
        /// <summary>
        /// Type of activation functions.
        /// </summary>
        public enum Type
        {
            None, Tanh, Sigmoid, Relu, SoftPlus, ReluLeaky
        }

        /// <summary>
        /// Type of this activation function.
        /// </summary>
        private Type _currentType;

        /// <summary>
        /// Type of this activation function.
        /// </summary>
        public Type CurrentType => _currentType;

        /// <summary>
        /// Leakage of Leaky Rectified linear unit.
        /// </summary>
        private static readonly double ReluLeakyLeakage = 0.01;

        /// <summary>
        /// Name for none.
        /// </summary>
        private static readonly string NameNone = "None";

        /// <summary>
        /// Name for hyperbolic tangent.
        /// </summary>
        private static readonly string NameTanh = "Hyperbolic tangent";

        /// <summary>
        /// Name for sigmoid.
        /// </summary>
        private static readonly string NameSigmoid = "Sigmoid";

        /// <summary>
        /// Name for Rectified linear unit.
        /// </summary>
        private static readonly string NameRelu = "Rectified linear unit";

        /// <summary>
        /// Namr for soft plus.
        /// </summary>
        private static readonly string NameSoftPlus = "Soft plus";

        /// <summary>
        /// Name for Leaky rectified linear unit.
        /// </summary>
        private static readonly string NameReluLeaky = "Leaky rectified linear unit";

        /// <summary>
        /// Name of the current activation function.
        /// </summary>
        private string _name;

        /// <summary>
        /// Name of the current activation function.
        /// </summary>
        public string Name => _name;

        /// <summary>
        /// Function for none.
        /// </summary>
        private static readonly DoubleFunction FunctionNone = new DoubleFunction(x => x);

        /// <summary>
        /// Function for hyperbolic tangent.
        /// </summary>
        private static readonly DoubleFunction FunctionTanh = new DoubleFunction(x => Math.Tanh(x));

        /// <summary>
        /// Function for sigmoid.
        /// </summary>
        private static readonly DoubleFunction FunctionSigmoid = new DoubleFunction(x => 1 / (1 + Math.Exp(-x)));

        /// <summary>
        /// Function for rectified linear unit.
        /// </summary>
        private static readonly DoubleFunction FunctionRelu = new DoubleFunction(x => x >= 0 ? x : 0.0);

        /// <summary>
        /// Function for soft plus.
        /// </summary>
        private static readonly DoubleFunction FunctionSoftPlus = new DoubleFunction(x => Math.Log(1 + Math.Exp(x)));

        /// <summary>
        /// Function for leaky rectified linear unit.
        /// </summary>
        private static readonly DoubleFunction FunctionReluLeaky = new DoubleFunction(x => x >= 0 ? x : ReluLeakyLeakage * x);

        /// <summary>
        /// Function.
        /// </summary>
        private DoubleFunction _function;

        /// <summary>
        /// Function.
        /// </summary>
        public DoubleFunction Function => _function;

        /// <summary>
        /// Prime for none.
        /// </summary>
        private static readonly DoubleFunction PrimeNone = new DoubleFunction(x => 1.0);

        /// <summary>
        /// Prime for hyperbolic tangent.
        /// </summary>
        private static readonly DoubleFunction PrimeTanh = new DoubleFunction(x => 1 - Math.Tanh(x) * Math.Tanh(x));

        /// <summary>
        /// Prime for sigmoid.
        /// </summary>
        private static readonly DoubleFunction PrimeSigmoid = new DoubleFunction(x => Math.Exp(-x) / Math.Pow(1 + Math.Exp(-x), 2.0));

        /// <summary>
        /// Prime for rectified linear unit.
        /// </summary>
        private static readonly DoubleFunction PrimeRelu = new DoubleFunction(x => x >= 0 ? 1.0 : 0.0);

        /// <summary>
        /// Prime for soft plus.
        /// </summary>
        private static readonly DoubleFunction PrimeSoftPlus = new DoubleFunction(x => 1 / (1 + Math.Exp(-x)));

        /// <summary>
        /// Prime for leaky rectified linear unit.
        /// </summary>
        private static readonly DoubleFunction PrimeReluLeaky = new DoubleFunction(x => x >= 0 ? 1.0 : ReluLeakyLeakage);

        /// <summary>
        /// Prime.
        /// </summary>
        private DoubleFunction _prime;  

        /// <summary>
        /// Prime.
        /// </summary>
        public DoubleFunction Prime => _prime;

        /// <summary>
        /// Returns current activation function's name.
        /// </summary>
        /// <returns>Name of the current activation function.</returns>
        public override string ToString()
        {
            return _name;
        }

        /// <summary>
        /// Changes the funciton type of this instance.
        /// </summary>
        /// <param name="type">The expected function type.</param>
        public void ChangeType(Type type)
        {
            switch (type)
            {
                case Type.None:
                    _name = NameNone;
                    _function = FunctionNone;
                    _prime = PrimeNone;
                    break;
                case Type.Tanh:
                    _name = NameTanh;
                    _function = FunctionTanh;
                    _prime = PrimeTanh;
                    break;
                case Type.Sigmoid:
                    _name = NameSigmoid;
                    _function = FunctionSigmoid;
                    _prime = PrimeSigmoid;
                    break;
                case Type.Relu:
                    _name = NameRelu;
                    _function = FunctionRelu;
                    _prime = PrimeRelu;
                    break;
                case Type.SoftPlus:
                    _name = NameSoftPlus;
                    _function = FunctionSoftPlus;
                    _prime = PrimeSoftPlus;
                    break;
                case Type.ReluLeaky:
                    _name = NameReluLeaky;
                    _function = FunctionReluLeaky;
                    _prime = PrimeReluLeaky;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, null);
            }
        }

        /// <summary>
        /// Constructs a new instance by given function type.
        /// </summary>
        /// <param name="type">The given function type.</param>
        public ActivationFunction(Type type)
        {
            ChangeType(type);
        }

        /// <summary>
        /// Constructs a new instance by given activation function's type.
        /// </summary>
        /// <param name="activationFunction"></param>
        public ActivationFunction(ActivationFunction activationFunction)
        {
            ChangeType(activationFunction._currentType);
        }
    }
}
