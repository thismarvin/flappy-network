
using System;

using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Media;

namespace ML.Engine.Resources
{
    static class Assets
    {
        public static Texture2D Text8x8 { get; private set; }

        public static void LoadContent(ContentManager Content)
        {
            Text8x8 = Content.Load<Texture2D>("Assets/Sprites/ASCII 8");
        }
    }
}
