
using System;
using System.Text;
using System.Collections.Generic;

using Microsoft.Xna.Framework.Graphics;

using ML.Engine.Resources;
using ML.Engine.GameComponents;
using System.Linq;

namespace ML.Engine.Utilities
{
    class Number : MonoObject
    {
        List<Sprite> sprites;
        public int CurrentValue { get; private set; }
        public int MaxValue { get; private set; }

        public Number(float x, float y, int currentValue, int totalPlaceValue, int maxValue, Sprite.Type whatFont) : base(x, y)
        {
            MaxValue = maxValue;
            CurrentValue = currentValue;

            sprites = new List<Sprite>();

            int textWidth = 0;
            switch (whatFont)
            {
                case Sprite.Type.Text8x8:
                    textWidth = 8;
                    break;
            }

            for (int i = totalPlaceValue - 1; i >= 0; i--)
            {
                sprites.Add(new Sprite((int)x + (textWidth * i), (int)y, whatFont));
            }

            Set(CurrentValue);
        }

        public void Increment(int incrementValue)
        {
            CurrentValue += incrementValue;

            if (CurrentValue > MaxValue)
            {
                CurrentValue = MaxValue;
            }

            Set(CurrentValue);
        }

        private int SetVauleFromString(string number)
        {
            return number.ToCharArray()[0] - 48;
        }

        public void Set(int value)
        {
            Reset();

            int test = value.ToString().Length - 1;
            for (int i = 0; i < sprites.Count; i++)
            {
                if (test >= 0)
                {
                    sprites[i].SetFrame(value.ToString().Substring(test, 1).ToCharArray()[0], 16);
                    test--;
                }
            }
            CurrentValue = value;
        }

        private void Reset()
        {
            foreach (Sprite s in sprites)
            {
                s.SetFrame("0".ToCharArray()[0], 16);
            }
        }

        public void Draw(SpriteBatch spriteBatch)
        {
            foreach (Sprite S in sprites)
            {
                S.Draw(spriteBatch);
            }
        }
    }
}
