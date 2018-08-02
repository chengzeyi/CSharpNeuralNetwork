using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text;

namespace CSharpNeuralNetwork
{
    /// <summary>
    /// Matrix class for storage calculation. This class can't be inherited.
    /// </summary>
    public sealed class Matrix
    {
        /// <summary>
        /// Matrix dimensions.
        /// </summary>
        private readonly int _height, _width;

        /// <summary>
        /// Matrix elements.
        /// </summary>
        private readonly double[,] _matrix;

        /// <summary>
        /// Number of Rows of the matrix.
        /// </summary>
        public int Height => _height;

        /// <summary>
        /// Number of columns of the matrix.
        /// </summary>
        public int Width => _width;

        /// <summary>
        /// Constructs a new matrix with the given dimensions.
        /// </summary>
        /// <param name="height">Number of rows.</param>
        /// <param name="width">Number of columns.</param>
        /// <exception cref="ArgumentException"></exception>
        public Matrix(int height, int width)
        {
            if (height < 1 || width < 1)
            {
                throw new ArgumentException("Dimension(s) less than 1!");
            }

            this._height = height;
            this._width = width;
            _matrix = new double[height, width];
        }

        /// <inheritdoc />
        /// <summary>
        /// Constructs a new copy of an existing matrix.
        /// </summary>
        /// <param name="input">Input matrix to copy.</param>
        public Matrix(Matrix input) : this(input._height, input._width)
        {
            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    _matrix[i, j] = input._matrix[i, j];
                }
            }
        }

        /// <inheritdoc />
        /// <summary>
        /// Constructs a new matrix with the content of a 2-dimensional array.
        /// </summary>
        /// <param name="array">Array to be copied into this matrix.</param>
        public Matrix(double[,] array) : this(array.GetUpperBound(0) + 1, array.GetLength(1))
        {
            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                   _matrix[i, j] = array[i, j];
                }
            }
        }

        /// <summary>
        /// Sets or returns the value of a specific element.
        /// </summary>
        /// <param name="row">Row index of the element.</param>
        /// <param name="column">Column index of the element.</param>
        /// <returns>The value of the element.</returns>
        public double this[int row, int column]
        {
            get => _matrix[row, column];
            set => _matrix[row, column] = value;
        }

        /// <summary>
        /// Sets every element of the matrix to the given value.
        /// </summary>
        /// <param name="value">Value Every element is set to.</param>
        public void Fill(double value)
        {
            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    _matrix[i, j] = value;
                }
            }
        }

        /// <summary>
        /// Fill the matirx with given random pattern and range.
        /// </summary>
        /// <param name="rand">The random pattern.</param>
        /// <param name="minimum">The minimum value.</param>
        /// <param name="maximum">The maximum value.</param>
        public void FillRand(Random rand, double minimum, double maximum)
        {
            double range = maximum - minimum;

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    _matrix[i, j] = range * rand.NextDouble() + minimum;
                }
            }
        }

        /// <summary>
        /// Fill the matrix with random values.
        /// </summary>
        public void FillRand()
        {
            FillRand(new Random(), -1, 1);
        }

        /// <summary>
        /// Adds two given matrix.
        /// </summary>
        /// <param name="lhs">The lest operand.</param>
        /// <param name="rhs">The right operand.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The sum of the two matrix.</returns>
        public static Matrix operator +(Matrix lhs, Matrix rhs)
        {
            if (lhs._height != rhs._height || lhs._width != rhs._width)
            {
                throw new ArgumentException("Dimensions not compatible!");
            }

            Matrix result = new Matrix(lhs._height, lhs._width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    result._matrix[i, j] = lhs._matrix[i, j] + rhs._matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Subtracts two given matrix.
        /// </summary>
        /// <param name="lhs">The lest operand.</param>
        /// <param name="rhs">The right operand.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The difference of the two matrix.</returns>
        public static Matrix operator -(Matrix lhs, Matrix rhs)
        {
            if (lhs._height != rhs._height || lhs._width != rhs._width)
            {
                throw new ArgumentException("Dimensions not compatible!");
            }

            Matrix result = new Matrix(lhs._height, lhs._width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    result._matrix[i, j] = lhs._matrix[i, j] - rhs._matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Multiplies two given matrix by elementwise.
        /// </summary>
        /// <param name="lhs">The lest operand.</param>
        /// <param name="rhs">The right operand.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The product of the two matrix.</returns>
        public static Matrix operator *(Matrix lhs, Matrix rhs)
        {
            if (lhs._height != rhs._height || lhs._width != rhs._width)
            {
                throw new ArgumentException("Dimensions not compatible!");
            }

            Matrix result = new Matrix(lhs._height, lhs._width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    result._matrix[i, j] = lhs._matrix[i, j] * rhs._matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Multiplies a matrix and a double value.
        /// </summary>
        /// <param name="lhs">The lest operand as a matrix.</param>
        /// <param name="rhs">The right operand as a double value.</param>
        /// <returns>The product.</returns>
        public static Matrix operator *(Matrix lhs, double rhs)
        {
            Matrix result = new Matrix(lhs._height, lhs._width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    result._matrix[i, j] = lhs._matrix[i, j] * rhs;
                }
            }

            return result;
        }

        /// <summary>
        /// Multiplies a double value and a matrix.
        /// </summary>
        /// <param name="lhs">The lest operand as a double value.</param>
        /// <param name="rhs">The right operand as a matrix.</param>
        /// <returns>The product.</returns>
        public static Matrix operator *(double lhs, Matrix rhs)
        {
            return rhs * lhs;
        }

        /// <summary>
        /// Divides two given matrix by elementwise.
        /// </summary>
        /// <param name="lhs">The lest operand.</param>
        /// <param name="rhs">The right operand.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The quotient of the two matrix.</returns>
        public static Matrix operator /(Matrix lhs, Matrix rhs)
        {
            if (lhs._height != rhs._height || lhs._width != rhs._width)
            {
                throw new ArgumentException("Dimensions not compatible!");
            }

            Matrix result = new Matrix(lhs._height, lhs._width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    result._matrix[i, j] = lhs._matrix[i, j] / rhs._matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Divides a matrix and a double value.
        /// </summary>
        /// <param name="lhs">The lest operand as a matrix.</param>
        /// <param name="rhs">The right operand as a double value.</param>
        /// <returns>The difference.</returns>
        public static Matrix operator /(Matrix lhs, double rhs)
        {
            Matrix result = new Matrix(lhs._height, lhs._width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    result._matrix[i, j] = lhs._matrix[i, j] / rhs;
                }
            }

            return result;
        }

        /// <summary>
        /// Returns the opposite matrix of the given one.
        /// </summary>
        /// <param name="matrix">The given matrix.</param>
        /// <returns>The opposite matrix.</returns>
        public static Matrix operator -(Matrix matrix)
        {
            Matrix result = new Matrix(matrix._height, matrix._width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    result._matrix[i, j] = -matrix._matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Returns a matrix equal to the given one.
        /// </summary>
        /// <param name="matrix">The given matrix.</param>
        /// <returns>The equal matrix.</returns>
        public static Matrix operator +(Matrix matrix)
        {
            Matrix result = new Matrix(matrix);

            return result;
        }

        /// <summary>
        /// Multiplies two given matrix.
        /// </summary>
        /// <param name="rhs">The right operand.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The product of the two matrix.</returns>
        public Matrix Dot(Matrix rhs)
        {
            if (_width != rhs._height)
            {
                throw new ArgumentException("Dimensions not compatible!");
            }

            Matrix result = new Matrix(_height, rhs._width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < _width; k++)
                    {
                        sum += _matrix[i, k] * rhs._matrix[k, j];
                    }

                    result._matrix[i, j] = sum; 
                }
            }

            return result;
        }

        /// <summary>
        /// Applies the given function to this matrix.
        /// </summary>
        /// <param name="function">The given function.</param>
        /// <returns>The result matrix.</returns>
        public Matrix Apply(DoubleFunction function)
        {
            Matrix result = new Matrix(_height, _width);

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result._matrix[i, j] = function(_matrix[i, j]);
                }
            }

            return result;
        }

        /// <summary>
        /// Applies the given function to this and another matrix.
        /// </summary>
        /// <param name="matrixB">Another matrix.</param>
        /// <param name="function">The given function.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The result matrix.</returns>
        public Matrix Apply(Matrix matrixB, BiDoubleFunction function)
        {
            if (_height != matrixB._height || _width != matrixB._width)
            {
                throw new ArgumentException("Dimensions not compatible!");
            }

            Matrix result = new Matrix(_height, _width);

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result._matrix[i, j] = function(_matrix[i, j], matrixB._matrix[i, j]);
                }
            }

            return result;
        }

        /// <summary>
        /// Transposes this matrix.
        /// </summary>
        /// <returns>The transposed matrix.</returns>
        public Matrix Transpose()
        {
            Matrix result = new Matrix(_width, _height);

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result._matrix[j, i] = _matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Extracts multiple rows as a new matrix.
        /// </summary>
        /// <param name="startIndex">The start row index.</param>
        /// <param name="count">The total row count.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <returns>The new matrix.</returns>
        public Matrix GetRows(int startIndex, int count)
        {
            if (count <= 0)
            {
                throw new ArgumentException("Count must be greater than zero!");
            }

            if (startIndex < 0 || startIndex >= _height || startIndex + count > _height)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            Matrix result = new Matrix(count, _width);

            for (int i = 0; i < result._height; i++)
            {
                for (int j = 0; j < result._width; j++)
                {
                    result._matrix[i, j] = _matrix[startIndex + i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Extracts a single row as a new matrix.
        /// </summary>
        /// <param name="index">The row index.</param>
        /// <returns>The new matrix.</returns>
        public Matrix GetRow(int index)
        {
            return GetRows(index, 1);
        }

        /// <summary>
        /// Appends a matrix to the bottom end of this matrix.
        /// </summary>
        /// <param name="rows">Rows to be appended with.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The merged matrix.</returns>
        public Matrix AppendRows(Matrix rows)
        {
            if (rows._width != _width)
            {
                throw new ArgumentException("Rows not compatible!");
            }

            Matrix result = new Matrix(_height + rows._height, _width);

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result._matrix[i, j] = _matrix[i, j];
                }
            }

            for (int i = 0; i < rows._height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result._matrix[i + _height, j] = rows._matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Removes multiple rows of this matrix.
        /// </summary>
        /// <param name="startIndex">The start row index.</param>
        /// <param name="count">The total row count.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <returns>The result matrix.</returns>
        public Matrix RemoveRows(int startIndex, int count)
        {
            if (count <= 0)
            {
                throw new ArgumentException("Count must be greater than zero!");
            }

            if (startIndex < 0 || startIndex >= _height || startIndex + count <= 0 || startIndex + count > _height)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            Matrix result = new Matrix(_height - count, _width);

            for (int i = 0; i < startIndex; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result._matrix[i, j] = _matrix[i, j];
                }
            }

            for (int i = 0; i < result._height - startIndex; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result._matrix[startIndex + i, j] = _matrix[startIndex + count + i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Remove a single row of this matrix.
        /// </summary>
        /// <param name="index">The row index.</param>
        /// <returns>The result matrix.</returns>
        public Matrix RemoveRow(int index)
        {
            return RemoveRows(index, 1);
        }

        /// <summary>
        /// Extracts multiple columns of this matrix.
        /// </summary>
        /// <param name="startIndex">The start index.</param>
        /// <param name="count">The total count.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <returns>The result matrix.</returns>
        public Matrix GetColumns(int startIndex, int count)
        {
            if (count <= 0)
            {
                throw new ArgumentException("Count must be greater than zero!");
            }

            if (startIndex < 0 || startIndex >= _width || startIndex + count <= 0 || startIndex + count > _width)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            Matrix result = new Matrix(_height, count);

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < count; j++)
                {
                    result._matrix[i, j] = _matrix[i, startIndex + j];
                }
            }

            return result;
        }

        /// <summary>
        /// Extracts a single column of this matrix.
        /// </summary>
        /// <param name="index">The column index.</param>
        /// <returns>The result matrix.</returns>
        public Matrix GetColumn(int index)
        {
            return GetColumns(index, 1);
        }

        /// <summary>
        /// Appends a matrix to the right end of this matrix.
        /// </summary>
        /// <param name="columns">The matrix to append with.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <returns>The result matrix.</returns>
        public Matrix AppendColumns(Matrix columns)
        {
            if (columns._height != _height)
            {
                throw new ArgumentException("Columns not compatible!");
            }

            Matrix result = new Matrix(_height, _width + columns._width);

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result._matrix[i, j] = _matrix[i, j];
                }

                for (int j = 0; j < columns._width; j++)
                {
                    result._matrix[i, _width + j] = columns._matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Removes multiple columns of this matrix.
        /// </summary>
        /// <param name="startIndex">The start column index.</param>
        /// <param name="count">The total column count.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <returns>The result matrix.</returns>
        public Matrix RemoveColumns(int startIndex, int count)
        {
            if (count <= 0)
            {
                throw new ArgumentException("Count must be greater than zero!");
            }

            if (startIndex < 0 || startIndex >= _width || startIndex + count < 0 || startIndex + count >= _width)
            {
                throw new IndexOutOfRangeException("Index out of range!");
            }

            Matrix result = new Matrix(_height, _width - count);

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < startIndex; j++)
                {
                    result._matrix[i, j] = _matrix[i, j];
                }

                for (int j = 0; j < result._width - startIndex; j++)
                {
                    result._matrix[i, startIndex + j] = _matrix[i, startIndex + count + j];
                }
            }

            return result;
        }

        /// <summary>
        /// Remove a single colunm of this matrix.
        /// </summary>
        /// <param name="index">The column index.</param>
        /// <returns>The result matrix.</returns>
        public Matrix RemoveColumn(int index)
        {
            return RemoveColumns(index, 1);
        }

        /// <summary>
        /// Copies the content of the matrix into a two-demensional array.
        /// </summary>
        /// <returns>The result array.</returns>
        public double[,] ToArray()
        {
            double[,] result = new double[_height, _width];

            for (int i = 0; i < _height; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    result[i, j] = _matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Returns the string form of this matrix.
        /// </summary>
        /// <returns>The string form of this matrix.</returns>
        public override string ToString()
        {
            StringBuilder builder = new StringBuilder("[[").Append(_matrix[0, 0]);

            for (int i = 1; i < _width; i++)
            {
                builder.Append(", ").Append(_matrix[0, i]);
            }

            builder.Append("]");

            for (int i = 1; i < _height; i++)
            {
                builder.Append("\n [").Append(_matrix[i, 0]);

                for (int j = 1; j < _width; j++)
                {
                    builder.Append(", ").Append(_matrix[i, j]);
                }

                builder.Append("]");
            }

            builder.Append("]");

            return builder.ToString();
        }
    }
}
