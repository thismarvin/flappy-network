
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Serialization;
using System.IO;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;

using ML.Engine.GameComponents;
using ML.Engine.Utilities;

namespace ML.Engine.Entities
{
    abstract class Entity : MonoObject, IComparable
    {
        public Vector2 Center { get; private set; }
        public Rectangle CollisionRectangle { get; private set; }
        public Rectangle ScaledCollisionRectangle { get { return new Rectangle(CollisionRectangle.X * Camera.Scale, CollisionRectangle.Y * Camera.Scale, Width * Camera.Scale, Height * Camera.Scale); } }
        public Color ObjectColor { get; set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int LayerDepth { get; set; }
        public bool Remove { get; set; }

        public Entity(float x, float y, int width, int height) : base(x, y)
        {
            ObjectColor = Color.White;
            Width = width;
            Height = height;
            LayerDepth = 1;
           
            SetCenter(X + Width / 2, Y + Height / 2);
            SetCollisionRectangle(X, Y, Width, Height);
        }

        public new void SetLocation(float x, float y)
        {
            base.SetLocation(x, y);
            SetCollisionRectangle(X, Y, Width, Height);
            SetCenter(X + Width / 2, Y + Height / 2);
        }

        public void SetCenter(float x, float y)
        {
            base.SetLocation(x - Width / 2, y - Height / 2);
            Center = new Vector2(X + Width / 2, Y + Height / 2);
        }

        public void SetCollisionRectangle(float x, float y, int width, int height)
        {
            base.SetLocation(x, y);
            Width = width;
            Height = height;
            CollisionRectangle = new Rectangle((int)X, (int)Y, Width, Height);
        }

        public void SetWidth(int width)
        {
            Width = width;
            SetCollisionRectangle(X, Y, Width, Height);
        }

        public void SetHeight(int height)
        {
            Height = height;
            SetCollisionRectangle(X, Y, Width, Height);
        }

        public void SetColor(Color color)
        {
            ObjectColor = color;
        }

        public int CompareTo(object obj)
        {
            return LayerDepth.CompareTo(((Entity)obj).LayerDepth);
        }

        public abstract void Update(GameTime gameTime);

        public abstract void Draw(SpriteBatch spriteBatch);
    }
}
