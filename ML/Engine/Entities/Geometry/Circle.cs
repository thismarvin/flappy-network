using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ML.Engine.GameComponents;
using ML.Engine.Utilities;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

namespace ML.Engine.Entities.Geometry
{
    class Circle : Entity
    {
        List<Line> lines;
        public int Radius { get; private set; }
        int lineWidth;
        const float INCREMENT = (float)Math.PI * 2 / 360;

        public Circle(float x, float y, int radius) : base(x, y, 1, 1)
        {
            lines = new List<Line>();
            Radius = radius;
            lineWidth = radius;

            CreateCircle(X, Y);
        }

        public Circle(float x, float y, int radius, int lineWidth) : this(x, y, radius)
        {
            this.lineWidth = lineWidth;
            CreateCircle(X, Y);
        }

        private void CreateCircle(float x, float y)
        {
            lines.Clear();
            for (float i = 0; i < Math.PI; i += INCREMENT / 2)
            {
                lines.Add(new Line(x - Radius + CircleX(i), y + CircleY(i), x - Radius + CircleX(i + INCREMENT), y + CircleY(i + INCREMENT), lineWidth, ObjectColor));
            }
        }

        public new void SetLocation(float x, float y)
        {
            base.SetLocation(x, y);
            CreateCircle(X, Y);
        }

        public new void SetCenter(float x, float y)
        {
            SetLocation(x, y);
        }

        public new void SetCollisionRectangle(float x, float y, int width, int height)
        {
            SetLocation(x, y);
        }

        public void SetLineWidth(int lineWidth)
        {
            this.lineWidth = lineWidth <= Radius ? lineWidth : Radius;
            CreateCircle(X, Y);
        }

        public void SetRadius(int radius)
        {
            Radius = radius;
            CreateCircle(X, Y);
        }

        private float CircleX(float x)
        {
            return (float)((Math.Cos(x)) * Math.Cos(x)) * Radius * 2;
        }

        private float CircleY(float y)
        {
            return (float)((Math.Cos(y)) * Math.Sin(y)) * Radius * 2;
        }

        public override void Draw(SpriteBatch spriteBatch)
        {
            foreach (Line l in lines)
            {
                l.Draw(spriteBatch);
            }

            if (Game1.DebugMode)
            {
                spriteBatch.Draw(ShapeManager.Texture, CollisionRectangle, Palette.TreeGreen);
            }
        }

        public override void Update(GameTime gameTime)
        {

        }
    }
}
