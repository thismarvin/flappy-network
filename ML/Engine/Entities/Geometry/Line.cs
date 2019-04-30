
using System;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

using ML.Engine.Utilities;

namespace ML.Engine.Entities.Geometry
{
    class Line : Entity
    {
        Vector2 origin;
        Vector2 endPoint;
        float rotation;
        float distance;
        int thickness;

        public Line(float x1, float y1, float x2, float y2) : base(x1, y1, 1, 1)
        {
            origin = new Vector2(x1, y1);
            endPoint = new Vector2(x2, y2);
            ObjectColor = Color.White;
            thickness = 1;

            Setup();
        }

        public Line(float x1, float y1, float x2, float y2, int thickness) : this(x1, y1, x2, y2)
        {
            this.thickness = thickness;
        }

        public Line(float x1, float y1, float x2, float y2, Color color) : this(x1, y1, x2, y2)
        {
            ObjectColor = color;
        }

        public Line(float x1, float y1, float x2, float y2, int thickness, Color color) : this(x1, y1, x2, y2)
        {
            this.thickness = thickness;
            ObjectColor = color;
        }

        private void Setup()
        {
            distance = (float)(Math.Sqrt(Math.Pow(Math.Abs(endPoint.X - origin.X), 2) + Math.Pow(Math.Abs(endPoint.Y - origin.Y), 2)));
            rotation = (float)Math.Atan2(endPoint.Y - origin.Y, endPoint.X - origin.X);
        }

        public void SetOrigin(float x, float y)
        {
            origin = new Vector2(x, y);
            Setup();
        }

        public void SetEndPoint(float x, float y)
        {
            endPoint = new Vector2(x, y);
            Setup();
        }

        public void SetThickness(int thickness)
        {
            this.thickness = thickness;
        }

        public override void Draw(SpriteBatch spriteBatch)
        {
            spriteBatch.Draw(ShapeManager.Texture, new Vector2(origin.X * Camera.Scale, origin.Y * Camera.Scale), null, ObjectColor, rotation, Vector2.Zero, new Vector2(distance * Camera.Scale, thickness * Camera.Scale), SpriteEffects.None, 0);
        }

        public override void Update(GameTime gameTime)
        {

        }
    }
}
