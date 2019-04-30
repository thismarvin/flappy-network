
using System;
using System.Collections.Generic;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

using ML.Engine.Utilities;

namespace ML.Engine.Entities.Geometry
{
    class Shape : Entity
    {
        List<Rectangle> addRectangles;
        int lineWidth;
        public bool Show { get; set; }
        public Tag ID { get; set; }

        public enum Tag
        {
            None
        }

        public Shape(float x, float y, int width, int height, Color objectColor) : base(x, y, width, height)
        {
            Show = true;
            ObjectColor = objectColor;
            addRectangles = new List<Rectangle>();
            ID = Tag.None;
            Setup();
        }

        public Shape(float x, float y, int width, int height, int lineWidth, Color objectColor) : this(x, y, width, height, objectColor)
        {
            this.lineWidth = lineWidth;
            Setup();
        }

        public Shape(float x, float y, int width, int height, int lineWidth, Tag tag, Color objectColor) : this(x, y, width, height, lineWidth, objectColor)
        {
            ID = tag;
        }

        private void Setup()
        {
            addRectangles.Clear();
            if (lineWidth == 0)
            {
                addRectangles.Add(new Rectangle((int)ScaledLocation.X, (int)ScaledLocation.Y, Width * Camera.Scale, Height * Camera.Scale));
            }
            else
            {
                addRectangles.Add(new Rectangle((int)ScaledLocation.X, (int)ScaledLocation.Y, Width * Camera.Scale, lineWidth * Camera.Scale));
                addRectangles.Add(new Rectangle((int)ScaledLocation.X, (int)ScaledLocation.Y + (Height - lineWidth) * Camera.Scale, Width * Camera.Scale, lineWidth * Camera.Scale));
                addRectangles.Add(new Rectangle((int)ScaledLocation.X, (int)ScaledLocation.Y + lineWidth * Camera.Scale, lineWidth * Camera.Scale, Height * Camera.Scale - lineWidth * 2 * Camera.Scale));
                addRectangles.Add(new Rectangle((int)ScaledLocation.X + Width * Camera.Scale - lineWidth * Camera.Scale, (int)ScaledLocation.Y + lineWidth * Camera.Scale, lineWidth * Camera.Scale, Height * Camera.Scale - lineWidth * 2 * Camera.Scale));
            }
        }

        public new void SetLocation(float x, float y)
        {
            base.SetLocation(x, y);
            Setup();
        }

        public new void SetCenter(float x, float y)
        {
            base.SetCenter(x, y);
            Setup();
        }

        public new void SetWidth(int width)
        {
            base.SetWidth(width);
            Setup();
        }

        public new void SetHeight(int height)
        {
            base.SetHeight(height);
            Setup();
        }

        public override void Draw(SpriteBatch spriteBatch)
        {
            if (Show)
            {
                foreach (Rectangle R in addRectangles)
                {
                    spriteBatch.Draw(ShapeManager.Texture, R, ObjectColor);
                }
            }
        }

        public override void Update(GameTime gameTime)
        {

        }
    }
}
