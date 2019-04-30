
using System;
using System.Collections.Generic;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

using ML.Engine.Utilities;

namespace ML.Engine.Entities
{
    abstract class Kinetic : Entity
    {
        public Vector2 Velocity { get; private set; }
        public List<Rectangle> CollisionRectangles { get; private set; }
        public List<Rectangle> ScaledCollisionRectangles { get; private set; }
        public float DefaultGravity { get; set; }
        public float Gravity { get; set; }
        public float MoveSpeed { private get; set; }
        protected float Speed { get; private set; }
        public int CollisionWidth { get; private set; }

        public enum Direction
        { Left, Right, Up, Down, None }
        public Direction Facing { get; set; }

        public bool Falling { get; set; }

        public Kinetic(float x, float y, int width, int height, float moveSpeed) : base(x, y, width, height)
        {
            Velocity = Vector2.Zero;
            DefaultGravity = 2.5f;
            MoveSpeed = moveSpeed;
            Facing = Direction.Right;
        }

        protected void SetVelocity(float x, float y)
        {
            Velocity = new Vector2(x, y);
        }

        protected void CalculateSpeed(GameTime gameTime)
        {
            Speed = MoveSpeed * (float)gameTime.ElapsedGameTime.TotalSeconds;
            Gravity = DefaultGravity * (float)gameTime.ElapsedGameTime.TotalSeconds;
        }

        protected abstract void ApplyForce();
        protected abstract void Collision();
    }
}
