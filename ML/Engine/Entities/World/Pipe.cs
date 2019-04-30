using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using ML.Engine.Entities.Geometry;
using ML.Engine.Level;
using ML.Engine.Utilities;

namespace ML.Engine.Entities.World
{
    class Pipe : Entity
    {
        public Shape Top { get; private set; }
        public Shape Bottom { get; private set; }

        public Shape topDecoration;
        public Shape bottomDecoration;

        public int OpeningHeight { get; private set; }

        public Pipe(float x) : base(x, 0, 32, 0)
        {
            OpeningHeight = (int)(Camera.ScreenBounds.Height * 0.2f);
            Top = new Shape(X, Y, Width, Playfield.RNG.Next(OpeningHeight, Camera.ScreenBounds.Height - OpeningHeight * 2), 2, Palette.GrassGreen);
            Bottom = new Shape(X, Y + Top.Height + OpeningHeight, Width, Camera.ScreenBounds.Height - OpeningHeight - Top.Height, 2, Palette.GrassGreen);
            topDecoration = new Shape(X - 2, Y + Top.Height - 16, Width + 4, 16, 2, Palette.GrassGreen);
            bottomDecoration = new Shape(X - 2, Y + Top.Height + OpeningHeight, Width + 4, 16, 2, Palette.GrassGreen);
        }

        public Pipe(float x, int height) : base(x, 0, 32, 0)
        {
            OpeningHeight = (int)(Camera.ScreenBounds.Height * 0.2f);
            Top = new Shape(X, Y, Width, height, 2, Palette.GrassGreen);
            Bottom = new Shape(X, Y + Top.Height + OpeningHeight, Width, Camera.ScreenBounds.Height - OpeningHeight - Top.Height, 2, Palette.GrassGreen);
            topDecoration = new Shape(X - 2, Y + Top.Height - 16, Width + 4, 16, 2, Palette.GrassGreen);
            bottomDecoration = new Shape(X - 2, Y + Top.Height + OpeningHeight, Width + 4, 16, 2, Palette.GrassGreen);
        }

        public bool Collidies(Entity e)
        {
            return e.CollisionRectangle.Intersects(Top.CollisionRectangle) || e.CollisionRectangle.Intersects(Bottom.CollisionRectangle);
        }

        public override void Update(GameTime gameTime)
        {

        }

        public override void Draw(SpriteBatch spriteBatch)
        {
            spriteBatch.Begin(SpriteSortMode.Deferred, BlendState.NonPremultiplied, SamplerState.PointClamp, null, null, null, Camera.Transform);
            {
                Top.Draw(spriteBatch);
                Bottom.Draw(spriteBatch);
                topDecoration.Draw(spriteBatch);
                bottomDecoration.Draw(spriteBatch);
            }
            spriteBatch.End();
        }
    }
}
