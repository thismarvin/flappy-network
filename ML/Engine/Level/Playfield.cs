using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

using ML.Engine.Utilities;
using ML.Engine.Entities;
using ML.Engine.Resources;
using ML.Engine.GameComponents;
using ML.Engine.Entities.Geometry;
using MathNet.Numerics.LinearAlgebra;
using ML.Engine.AI;
using ML.Engine.Entities.World;

namespace ML.Engine.Level
{
    static class Playfield
    {
        public static List<Entity> Entities { get; private set; }
        public static List<Entity> EntityBuffer { get; set; }
        public static List<Player> Players { get; private set; }
        public static Vector2 CameraLocation { get; private set; }
        public static Random RNG { get; set; }

        static bool noSpam;
        static int totalPlayers;
        static int generations;
        static int pipeIndex;
        static double furthestDistance;
        static List<NeuralNetwork> neuralNetworks;
        static Player best;
        static Player furthest;

        static Text generationsText;
        static Number generationsNumber;

        static Shape furthestLine;

        public static void Initialize()
        {
            Entities = new List<Entity>();
            EntityBuffer = new List<Entity>();
            Players = new List<Player>();
            neuralNetworks = new List<NeuralNetwork>();
            RNG = new Random(DateTime.Now.Millisecond);

            Reset();
        }

        public static void Reset()
        {
            totalPlayers = 500;
            generations = 1;
            furthestDistance = 0;

            generationsText = new Text(10, 10, "Generations", Sprite.Type.Text8x8);
            generationsNumber = new Number(10, 18, generations, 5, 99999, Sprite.Type.Text8x8);
            furthestLine = new Shape((float)furthestDistance, 0, 1, Camera.ScreenBounds.Height, Palette.BloodRed);

            Entities.Clear();
            ResetPlayers();
            ResetPipes();
        }

        private static void ResetPipes()
        {
            pipeIndex = 0;
            Entities.Add(new Pipe(Camera.ScreenBounds.Width * 0.5f + pipeIndex++ * 128, 128));
            Entities.Add(new Pipe(Camera.ScreenBounds.Width * 0.5f + pipeIndex++ * 128, 128));
            Entities.Add(new Pipe(Camera.ScreenBounds.Width * 0.5f + pipeIndex++ * 128, 96));
            Entities.Add(new Pipe(Camera.ScreenBounds.Width * 0.5f + pipeIndex++ * 128, 64));
            Entities.Add(new Pipe(Camera.ScreenBounds.Width * 0.5f + pipeIndex++ * 128, 64));
            Entities.Add(new Pipe(Camera.ScreenBounds.Width * 0.5f + pipeIndex++ * 128, 128));
        }

        private static void ResetPlayers()
        {
            Players.Clear();

            for (int i = 0; i < totalPlayers; i++)
                Players.Add(new Player(Camera.ScreenBounds.Width * 0.25f, Camera.ScreenBounds.Height * 0.5f));

            // Crude functionality to load a decent saved Neural Network
            // NeuralNetwork saved = new NeuralNetwork(
            // new double[,] {
            // { 0.56505161911982, 0.231433888898287, -0.499290977012648, -0.843156239517252 },
            // { -0.534387345781411, -0.928998347911198, 0.772845990277608, 0.361129986183632 },
            // { -0.961490761111866, 0.98325842467786, 1.00058302132937, 0.322531978643135 },
            // { 0.731571862707965, -0.499495992776384, -0.83061042011712, -0.113941927600306 },
            // { -0.0571549200855521, 0.741312287054955, -0.941430100784984, 0.527704262313614 },
            // { -0.791113907946473, -0.779075292408648, 0.292312881007324, 0.318368697580441 },
            // },
            // new double[,] {
            // { -0.37006703764985, -0.705871440634437, -0.637093318758476, -0.182028305637098, 0.391772774395061, 0.786749203288911 },
            // { -0.412795481299786, 0.0489219475547487, 0.316330861612011, 0.00331542657526673, 0.897934738197899, 0.826042396565553 },
            // },
            // new double[,] {
            // { -0.211874674761109 },
            // { -0.637768462254057 },
            // { -0.690330257617522 },
            // { 0.841617328849944 },
            // { 0.54273099232456 },
            // { 0.125569206132853 },
            // },
            // new double[,] {
            // { 0.97682700014982 },
            // { 0.00789955573980901 },
            // },
            // ActivationFunction.Functions.Sigmoid
            // );
 
            //Players.Add(new Player(Camera.ScreenBounds.Width * 0.25f, Camera.ScreenBounds.Height * 0.5f, saved));

            foreach (Player p in Players)
            {
                Entities.Add(p);
            }
        }

        private static Player FurthestPlayer()
        {
            Player result = Players[0];
            foreach (Player p in Players)
            {
                if (p.CollisionRectangle.Right > result.CollisionRectangle.Right)
                {
                    result = p;
                }
            }
            return result;
        }

        private static void CalculateFitness()
        {
            double sum = 0;
            foreach (Player p in Players)
            {
                p.Score = Math.Pow(p.DefaultScore, 3);
                sum += p.Score;
            }
            foreach (Player p in Players)
            {
                p.Fitness = p.Score / sum;
            }
            best = PlayerWithHighestFitness();
        }

        private static Player PlayerWithHighestFitness()
        {
            Player best = Players[0];
            foreach (Player p in Players)
            {
                if (p.Fitness > best.Fitness)
                {
                    best = p;
                }
            }
            return best;
        }

        private static Player RandomSelection()
        {
            int index = 0;
            double random = RNG.NextDouble();

            while (random > 0)
            {
                random -= Players[index].Fitness;
                index++;
            }
            index--;
            return Players[index];
        }

        private static bool PlayersAreStillAlive()
        {
            foreach (Player p in Players)
            {
                if (!p.Finished)
                {
                    return true;
                }
            }
            return false;
        }

        private static void GeneticAlgorithm()
        {
            // Create the next generation once all the players are dead.
            if (PlayersAreStillAlive())
                return;

            // Setup the Neural Networks for the next generation.
            neuralNetworks.Clear();
            // Copy the best player's neural network and modify its weights slightly.
            /// This guarantees that the best player will be in the next generation and not randomly die out.
            for (int i = 0; i < (int)(totalPlayers * 0.1); i++)
            {
                neuralNetworks.Add(new NeuralNetwork(best.Brain));
                neuralNetworks.Last().Mutate(0.50, 0.1);
            }
            // Copy the best player's neural network and modify its weights more drastically and more frequently.
            /// This will add more variance to the best player in hopes to make it even better.
            for (int i = 0; i < (int)(totalPlayers * 0.4); i++)
            {
                neuralNetworks.Add(new NeuralNetwork(best.Brain));
                neuralNetworks.Last().Mutate(0.75, 1);
            }
            // Randomly select players based on their fitness score and modify their weights very drastically and very frequently.
            /// In this stage, any player has a chance of overtaking the best player purely based on RNG.
            for (int i = 0; i < (int)(totalPlayers * 0.4); i++)
            {
                neuralNetworks.Add(new NeuralNetwork(RandomSelection().Brain));
                neuralNetworks.Last().Mutate(0.90, 2);
            }
            // Create completely random players.
            /// This step is included just in case the trained players reach a wall, figuratively, and stop improving entirely.
            for (int i = 0; i < (int)(totalPlayers * 0.1); i++)
            {
                neuralNetworks.Add(new NeuralNetwork(4, 6, 2));
            }

            Players.Clear();
            Entities.Clear();
            // Create the next generation of players and assign them a neural network.
            for (int i = 0; i < totalPlayers; i++)
            {
                Players.Add(new Player(Camera.ScreenBounds.Width * 0.25f, Camera.ScreenBounds.Height * 0.5f, neuralNetworks[i]));
                Entities.Add(Players.Last());
            }

            // Reset the world.
            best = Players[0];
            ResetPipes();

            generationsNumber.Set(++generations);
        }

        private static void BackToMenu()
        {
            Reset();
            Game1.GameMode = Game1.Mode.Menu;
        }

        private static void CameraHandler(GameTime gameTime)
        {
            CameraLocation = new Vector2(furthest.X - Camera.ScreenBounds.Width * 0.5f, 0);
            Camera.Update(CameraLocation, 0, Camera.ScreenBounds.Width * 10000, 0, Camera.ScreenBounds.Height);
        }

        private static void UpdateInput()
        {
            if (Keyboard.GetState().IsKeyDown(Keys.R) && noSpam)
            {
                Reset();
                noSpam = false;
            }
            if (Keyboard.GetState().IsKeyDown(Keys.Space) && noSpam)
            {
                Console.Clear();
                best.Brain.Save();
                noSpam = false;
            }
            noSpam = Keyboard.GetState().IsKeyUp(Keys.R) && Keyboard.GetState().IsKeyUp(Keys.Space) ? true : noSpam;
        }

        private static void UpdateVisualization()
        {           
            furthest = FurthestPlayer();
            furthestDistance = furthest.CollisionRectangle.Right > furthestDistance ? furthest.CollisionRectangle.Right : furthestDistance;
            furthestLine.SetLocation((float)furthestDistance, 0);
        }

        private static void UpdateEntities(GameTime gameTime)
        {
            for (int i = Entities.Count - 1; i >= 0; i--)
            {
                Entities[i].Update(gameTime);

                if (Entities[i] is Pipe)
                {
                    if (Entities[i].CollisionRectangle.Right < CameraLocation.X)
                    {
                        Entities[i].Remove = true;
                        EntityBuffer.Add(new Pipe(Camera.ScreenBounds.Width * 0.5f + pipeIndex++ * 128));
                    }
                }

                if (Entities[i].Remove)
                    Entities.RemoveAt(i);
            }

            if (EntityBuffer.Count > 0)
            {
                foreach (Entity e in EntityBuffer)
                    Entities.Add(e);

                EntityBuffer.Clear();
            }
        }

        public static void Update(GameTime gameTime)
        {
            UpdateInput();

            CalculateFitness();
            GeneticAlgorithm();

            UpdateVisualization();
            CameraHandler(gameTime);
            UpdateEntities(gameTime);
        }

        public static void Draw(SpriteBatch spriteBatch)
        {
            foreach (Entity e in Entities)
            {
                e.Draw(spriteBatch);
            }

            spriteBatch.Begin(SpriteSortMode.Deferred, BlendState.NonPremultiplied, SamplerState.PointClamp, null, null, null, Camera.Transform);
            {
                furthestLine.Draw(spriteBatch);
            }
            spriteBatch.End();

            spriteBatch.Begin(SpriteSortMode.Deferred, BlendState.NonPremultiplied, SamplerState.PointClamp, null, null, null, StaticCamera.Transform);
            {
                generationsText.Draw(spriteBatch);
                generationsNumber.Draw(spriteBatch);
            }
            spriteBatch.End();

            best.DrawNeuralNetwork(spriteBatch);
        }
    }
}
