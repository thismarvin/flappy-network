
using System;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

namespace ML.Engine.Utilities
{
    class ShapeManager
    {
        public static Texture2D Texture { get; private set; }

        private ShapeManager()
        {
            
        }

        public static void Initialize(GraphicsDeviceManager graphics)
        {
            Texture = new Texture2D(graphics.GraphicsDevice, 1, 1);
            Texture.SetData(new[] { Color.White });
        }

        public static void UnloadContent()
        {
            Texture.Dispose();
        }
    }
}
