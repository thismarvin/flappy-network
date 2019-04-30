using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using ML.Engine.Entities;
using ML.Engine.Entities.Geometry;
using ML.Engine.Level;
using ML.Engine.Utilities;

namespace ML.Engine.AI
{
    class NeuralNetwork
    {
        public int TotalInputNodes { get; private set; }
        public int TotalHiddenNodes { get; private set; }
        public int TotalOutputNodes { get; private set; }

        public Matrix<double> WeightsInputToHidden { get; private set; }
        public Matrix<double> WeightsHiddenToOutput { get; private set; }
        public Matrix<double> BiasHidden { get; private set; }
        public Matrix<double> BiasOutput { get; private set; }

        public Matrix<double> Input { get; private set; }
        public Matrix<double> Hidden { get; private set; }
        public Matrix<double> Output { get; private set; }

        public ActivationFunction ActivationFunction { get; private set; }

        List<Entity> entities;

        public NeuralNetwork(int totalInputNodes, int totalHiddenNodes, int totalOutputNodes)
        {
            TotalInputNodes = totalInputNodes;
            TotalHiddenNodes = totalHiddenNodes;
            TotalOutputNodes = totalOutputNodes;

            WeightsInputToHidden = Matrix<double>.Build.Dense(TotalHiddenNodes, TotalInputNodes, (i, j) => RandomRange(-1, 1));
            WeightsHiddenToOutput = Matrix<double>.Build.Dense(TotalOutputNodes, TotalHiddenNodes, (i, j) => RandomRange(-1, 1));
            BiasHidden = Matrix<double>.Build.Dense(TotalHiddenNodes, 1, (i, j) => RandomRange(-1, 1));
            BiasOutput = Matrix<double>.Build.Dense(TotalOutputNodes, 1, (i, j) => RandomRange(-1, 1));

            ActivationFunction = new ActivationFunction(ActivationFunction.Functions.Sigmoid);

            VisualSetup();
        }

        public NeuralNetwork(int totalInputNodes, int totalHiddenNodes, int totalOutputNodes, ActivationFunction.Functions function) : this(totalInputNodes, totalHiddenNodes, totalOutputNodes)
        {
            ActivationFunction = new ActivationFunction(function);
        }

        public NeuralNetwork(double[,] WeightsInputToHidden, double[,] WeightsHiddenToOutput, double[,] BiasHidden, double[,] BiasOutput, ActivationFunction.Functions function)
        {
            TotalInputNodes = WeightsInputToHidden.GetLength(1);
            TotalHiddenNodes = WeightsInputToHidden.GetLength(0);
            TotalOutputNodes = WeightsHiddenToOutput.GetLength(0);

            this.WeightsInputToHidden = Matrix<double>.Build.DenseOfArray(WeightsInputToHidden);
            this.WeightsHiddenToOutput = Matrix<double>.Build.DenseOfArray(WeightsHiddenToOutput);
            this.BiasHidden = Matrix<double>.Build.DenseOfArray(BiasHidden);
            this.BiasOutput = Matrix<double>.Build.DenseOfArray(BiasOutput);

            ActivationFunction = new ActivationFunction(function);

            VisualSetup();
        }

        public NeuralNetwork(NeuralNetwork neuralNetwork)
        {
            TotalInputNodes = neuralNetwork.TotalInputNodes;
            TotalHiddenNodes = neuralNetwork.TotalHiddenNodes;
            TotalOutputNodes = neuralNetwork.TotalOutputNodes;

            WeightsInputToHidden = neuralNetwork.WeightsInputToHidden;
            WeightsHiddenToOutput = neuralNetwork.WeightsHiddenToOutput;
            BiasHidden = neuralNetwork.BiasHidden;
            BiasOutput = neuralNetwork.BiasOutput;

            ActivationFunction = neuralNetwork.ActivationFunction;

            VisualSetup();
        }

        private void VisualSetup()
        {
            entities = new List<Entity>();
            Vector2 topLeft = new Vector2(50, 100);

            // Draw Nodes.
            int radius = 7;
            int spacing = 25;

            int inputHeight = TotalInputNodes * radius * 2 + (TotalInputNodes - 1) * spacing / 2;
            int hiddenHeight = TotalHiddenNodes * radius * 2 + (TotalHiddenNodes - 1) * spacing / 2;
            int outputHeight = TotalOutputNodes * radius * 2 + (TotalOutputNodes - 1) * spacing / 2;
            int offset = 0;

            for (int i = 0; i < TotalInputNodes; i++)
            {
                entities.Add(new Circle(topLeft.X + spacing * 0 * 2, topLeft.Y + i * (radius * 2 + spacing / 2) + offset, radius, 2));
            }

            offset = (inputHeight - hiddenHeight) / 2;
            for (int i = 0; i < TotalHiddenNodes; i++)
            {
                entities.Add(new Circle(topLeft.X + spacing * 1 * 2, topLeft.Y + i * (radius * 2 + spacing / 2) + offset, radius, 2));
            }

            offset = (inputHeight - outputHeight) / 2;
            for (int i = 0; i < TotalOutputNodes; i++)
            {
                entities.Add(new Circle(topLeft.X + spacing * 2 * 2, topLeft.Y + i * (radius * 2 + spacing / 2) + offset, radius, 2));
            }

            foreach (Entity e in entities)
            {
                if (e.LayerDepth == 1)
                {
                    e.LayerDepth = 3;
                }
            }

            // Draw Lines.
            Color color = Color.Blue;
            for (int i = 0; i < TotalInputNodes; i++)
            {
                for (int j = 0; j < TotalHiddenNodes; j++)
                {
                    color = WeightsInputToHidden[j, i] > 0 ? Color.Blue : Color.Red;
                    color = new Color(color.R, color.G, color.B, (int)(Math.Abs(WeightsInputToHidden[j, i]) * 230) + 25);
                    entities.Add(new Line(entities[i].Center.X, entities[i].Center.Y, entities[TotalInputNodes + j].Center.X, entities[TotalInputNodes + j].Center.Y, color));
                }
            }

            for (int i = 0; i < TotalHiddenNodes; i++)
            {
                for (int j = 0; j < TotalOutputNodes; j++)
                {
                    color = WeightsHiddenToOutput[j, i] > 0 ? Color.Blue : Color.Red;
                    color = new Color(color.R, color.G, color.B, (int)(Math.Abs(WeightsHiddenToOutput[j, i]) * 230) + 25);
                    entities.Add(new Line(entities[TotalInputNodes + i].Center.X, entities[TotalInputNodes + i].Center.Y, entities[TotalInputNodes + TotalHiddenNodes + j].Center.X, entities[TotalInputNodes + TotalHiddenNodes + j].Center.Y, color));
                }
            }

            foreach (Entity e in entities)
            {
                if (e.LayerDepth == 1)
                {
                    e.LayerDepth = 2;
                }
            }

            entities.Sort();
        }

        private void UpdateVisualization()
        {
            Color color = Color.Blue;
            for (int i = 0; i < TotalInputNodes; i++)
            {
                for (int j = 0; j < TotalHiddenNodes; j++)
                {
                    color = WeightsInputToHidden[j, i] > 0 ? Color.Blue : Color.Red;
                    color = new Color(color.R, color.G, color.B, (int)(Math.Abs(WeightsInputToHidden[j, i]) * 230) + 25);
                    entities[i + 1 + i * TotalHiddenNodes + j].SetColor(color);
                }
            }

            for (int i = 0; i < TotalHiddenNodes; i++)
            {
                for (int j = 0; j < TotalOutputNodes; j++)
                {
                    color = WeightsHiddenToOutput[j, i] > 0 ? Color.Blue : Color.Red;
                    color = new Color(color.R, color.G, color.B, (int)(Math.Abs(WeightsHiddenToOutput[j, i]) * 230) + 25);
                    entities[TotalInputNodes + TotalInputNodes * TotalHiddenNodes + i + 1 + i * TotalOutputNodes + j].SetColor(color);
                }
            }
        }

        private double RandomRange(double lowerBound, double upperBound)
        {
            return lowerBound + Playfield.RNG.NextDouble() * (upperBound - lowerBound);
        }

        private double RandomGaussian(double mean, double standardDeviation)
        {
            double u1 = 1.0 - Playfield.RNG.NextDouble();
            double u2 = 1.0 - Playfield.RNG.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + standardDeviation * randStdNormal;
        }

        private void ApplyActivationFunction(Matrix<double> matrix)
        {
            for (int y = 0; y < matrix.RowCount; y++)
            {
                for (int x = 0; x < matrix.ColumnCount; x++)
                {
                    matrix[y, x] = ActivationFunction.Compute(matrix[y, x]);
                }
            }
        }

        public Matrix<double> Predict(double[,] inputArray)
        {
            Input = Matrix<double>.Build.DenseOfArray(inputArray);

            Hidden = WeightsInputToHidden * Input;
            Hidden = Hidden.Add(BiasHidden);
            ApplyActivationFunction(Hidden);

            Output = WeightsHiddenToOutput * Hidden;
            Output = Output.Add(BiasOutput);
            ApplyActivationFunction(Output);

            return Output;
        }

        public void Mutate(double probability, double standardDeviation)
        {
            if (Playfield.RNG.NextDouble() < probability)
            {
                WeightsInputToHidden = WeightsInputToHidden.Add(RandomGaussian(0, standardDeviation));
            }
            if (Playfield.RNG.NextDouble() < probability)
            {
                WeightsHiddenToOutput = WeightsHiddenToOutput.Add(RandomGaussian(0, standardDeviation));
            }
            if (Playfield.RNG.NextDouble() < probability)
            {
                BiasHidden = BiasHidden.Add(RandomGaussian(0, standardDeviation));
            }
            if (Playfield.RNG.NextDouble() < probability)
            {
                BiasOutput = BiasOutput.Add(RandomGaussian(0, standardDeviation));
            }

            UpdateVisualization();
        }

        public void Save()
        {
            //Console.WriteLine("{0} {1} {2}", TotalInputNodes, TotalHiddenNodes, TotalOutputNodes);
            //Console.WriteLine("Weights Input to Hidden:");
            //Console.WriteLine(WeightsInputToHidden);
            //Console.WriteLine("Weights Hidden to Output:");
            //Console.WriteLine(WeightsHiddenToOutput);
            //Console.WriteLine("Bias Hidden:");
            //Console.WriteLine(BiasHidden);
            //Console.WriteLine("Bias Output:");
            //Console.WriteLine(BiasOutput);
            //Console.WriteLine("Activation Function:");
            //Console.WriteLine(ActivationFunction.Function);

            Console.WriteLine("NeuralNetwork saved = new NeuralNetwork(");
            Console.WriteLine("new double[,] {");
            for (int i = 0; i < TotalHiddenNodes; i++)
            {
                for (int j = 0; j < TotalInputNodes; j++)
                {
                    if (j == 0)
                        Console.Write("{ " + WeightsInputToHidden[i, j] + ", ");
                    else if (j < TotalInputNodes - 1)
                        Console.Write(WeightsInputToHidden[i, j] + ", ");
                    else
                        Console.Write(WeightsInputToHidden[i, j] + " },");
                }
                Console.WriteLine();
            }
            Console.WriteLine("},");

            Console.WriteLine("new double[,] {");
            for (int i = 0; i < TotalOutputNodes; i++)
            {
                for (int j = 0; j < TotalHiddenNodes; j++)
                {
                    if (j == 0)
                        Console.Write("{ " + WeightsHiddenToOutput[i, j] + ", ");
                    else if (j < TotalHiddenNodes - 1)
                        Console.Write(WeightsHiddenToOutput[i, j] + ", ");
                    else
                        Console.Write(WeightsHiddenToOutput[i, j] + " },");
                }
                Console.WriteLine();
            }
            Console.WriteLine("},");

            Console.WriteLine("new double[,] {");
            for (int i = 0; i < TotalHiddenNodes; i++)
            {
                Console.WriteLine("{ " + BiasHidden[i, 0] + " },");
            }
            Console.WriteLine("},");

            Console.WriteLine("new double[,] {");
            for (int i = 0; i < TotalOutputNodes; i++)
            {
                Console.WriteLine("{ " + BiasOutput[i, 0] + " },");
            }
            Console.WriteLine("},");

            Console.WriteLine("ActivationFunction.Functions.{0}", ActivationFunction.Function);
            Console.WriteLine("};");
        }

        public void Draw(SpriteBatch spriteBatch)
        {
            spriteBatch.Begin(SpriteSortMode.Deferred, BlendState.NonPremultiplied, SamplerState.PointClamp, null, null, null, StaticCamera.Transform);
            {
                foreach (Entity e in entities)
                {
                    e.Draw(spriteBatch);
                }
            }
            spriteBatch.End();
        }
    }
}
