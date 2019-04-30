using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using ML.Engine.AI;
using ML.Engine.Entities.Geometry;
using ML.Engine.Entities.World;
using ML.Engine.GameComponents;
using ML.Engine.Level;
using ML.Engine.Resources;
using ML.Engine.Utilities;

namespace ML.Engine.Entities
{
    class Player : Kinetic
    {
        public PlayerIndex PlayerIndex { get; private set; }
        Input input;

        public bool Dead { get; private set; }
        public bool Finished { get; private set; }

        bool grounded;

        float defaultJumpHeight;
        float jumpHeight;

        Pipe closestPipe;
        float topPipeBottom;
        float bottomPipeTop;
        float distanceToPipe;

        double[,] inputArray;
        Matrix<double> output;
        public NeuralNetwork Brain { get; private set; }
        public double Score { get; set; }
        public double DefaultScore { get; set; }
        public double Fitness { get; set; }

        public Player(float x, float y) : base(x, y, 12, 12, 1)
        {
            input = new Input(PlayerIndex.One);

            MoveSpeed = 50;

            defaultJumpHeight = 75;
            ObjectColor = new Color(255, 255, 255, 50);

            Brain = new NeuralNetwork(4, 6, 2, ActivationFunction.Functions.ReLU);
        }

        public Player(float x, float y, NeuralNetwork neuralNetwork) : base(x, y, 12, 12, 1)
        {
            input = new Input(PlayerIndex.One);

            MoveSpeed = 50;

            defaultJumpHeight = 75;
            ObjectColor = new Color(255, 255, 255, 50);
            Brain = neuralNetwork;
        }

        protected new void CalculateSpeed(GameTime gameTime)
        {
            base.CalculateSpeed(gameTime);
            jumpHeight = defaultJumpHeight * (float)gameTime.ElapsedGameTime.TotalSeconds;
        }

        public new void SetLocation(float x, float y)
        {
            base.SetLocation(x, y);
        }

        public new void SetCenter(float x, float y)
        {
            base.SetCenter(x, y);
            SetLocation(X - Width / 2, Y - Height / 2);
        }

        protected override void ApplyForce()
        {
            if (!Dead)
            {
                SetVelocity(Speed, Velocity.Y + Gravity);
            }
            else if (Dead && !grounded)
            {
                SetVelocity(0, Velocity.Y + Gravity * 3);
            }
            else
            {
                SetVelocity(0, 0);
            }

            SetLocation(Location.X + Velocity.X, Location.Y + Velocity.Y);
        }

        protected override void Collision()
        {
            if (!Dead)
            {
                foreach (Entity e in Playfield.Entities)
                {
                    if (e != this)
                    {
                        if (e is Pipe)
                        {
                            if (((Pipe)e).Collidies(this))
                            {
                                Dead = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (Location.Y > Camera.ScreenBounds.Height * 0.99f)
            {
                Dead = true;
                SetLocation(Location.X, Camera.ScreenBounds.Height * 0.99f - Height);
                grounded = true;
            }
            if (Location.Y < 0)
            {
                Dead = true;
                SetLocation(Location.X, 0);
                SetVelocity(0, 0);
            }
        }

        private Pipe ClosestPipe()
        {
            Pipe closest = null;
            foreach (Entity e in Playfield.Entities)
            {
                if (e is Pipe)
                {
                    if (closest == null ||
                        (
                        ((Pipe)e).Top.CollisionRectangle.Right > CollisionRectangle.Right &&
                        ((Pipe)e).Top.CollisionRectangle.Right - CollisionRectangle.Right < closest.Top.CollisionRectangle.Right - CollisionRectangle.Right
                        )
                       )
                    {
                        closest = (Pipe)e;
                    }

                    if (closest.Top.CollisionRectangle.Right < CollisionRectangle.Right)
                    {
                        closest = null;
                    }
                }
            }
            return closest;
        }

        private void UpdateInput(GameTime gameTime)
        {
            if (Dead)
                return;

            //input.Update(gameTime);
            //Facing = Direction.None;

            //if (input.Pressing(Input.InputType.MovementUp) && input.KeyReleased)
            //{
            //    Jump();
            //    input.KeyReleased = false;
            //}

            closestPipe = ClosestPipe();
            if (closestPipe != null)
            {
                distanceToPipe = Vector2.Distance(Center, new Vector2(closestPipe.X + closestPipe.Width, closestPipe.Top.CollisionRectangle.Bottom + closestPipe.OpeningHeight / 2));
                topPipeBottom = closestPipe.Top.CollisionRectangle.Bottom;
                bottomPipeTop = closestPipe.Top.CollisionRectangle.Top;
            }
            else
            {
                Console.WriteLine("ERROR");
            }

            inputArray = new double[,] { { Center.Y / Camera.ScreenBounds.Height }, { distanceToPipe / Camera.ScreenBounds.Width }, { topPipeBottom / Camera.ScreenBounds.Height }, { bottomPipeTop / Camera.ScreenBounds.Height } };

            output = Brain.Predict(inputArray);

            if (output[0, 0] > output[1, 0])
            {
                Jump();
            }
        }

        private void Jump()
        {
            SetVelocity(Velocity.X, -jumpHeight);
        }

        private void CalculateScore()
        {
            // This is very important to ensure that the next generation of players actually improves.
            // A lot of consideration needs to go into how the player is scored for its performance.
            /// In this case, the player's score is calculated by simply adding the distance the player traveled and the inverse of how far it was from the center of the pipe's opening when it died.
            Score = CollisionRectangle.Right + 1000 / distanceToPipe;
            DefaultScore = Score;
        }

        public override void Update(GameTime gameTime)
        {
            if (!Finished)
            {
                CalculateSpeed(gameTime);
                ApplyForce();
                UpdateInput(gameTime);
                CalculateScore();
            }

            Collision();

            Finished = Dead && grounded;
        }

        public void DrawNeuralNetwork(SpriteBatch spriteBatch)
        {
            Brain.Draw(spriteBatch);
        }

        public override void Draw(SpriteBatch spriteBatch)
        {
            spriteBatch.Begin(SpriteSortMode.Deferred, BlendState.NonPremultiplied, SamplerState.PointClamp, null, null, null, Camera.Transform);
            {
                spriteBatch.Draw(ShapeManager.Texture, ScaledCollisionRectangle, ObjectColor);
            }
            spriteBatch.End();
        }
    }
}
