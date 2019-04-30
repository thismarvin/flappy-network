
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

using ML.Engine.Resources;
using ML.Engine.GameComponents;

namespace ML.Engine.Utilities
{
    class Text : MonoObject
    {
        List<string> words;
        List<Sprite> sprites;
        Timer timer;
        Sprite.Type textType;
        float textWidth;
        int maximumCharacterCount;
        float spacing;
        public string Dialogue { get; private set; }
        bool showAll;
        float scale;    
        bool compact;

        public Text(float x, float y, string text, Sprite.Type type) : base(x, y)
        {
            words = new List<string>();
            sprites = new List<Sprite>();
            timer = new Timer(0);

            Dialogue = text;
            textType = type;
            maximumCharacterCount = text.Length * 2;
            compact = true;
            showAll = true;
            scale = 1;

            BreakUpWords();
            CreateText();
        }

        public Text(float x, float y, string text, Sprite.Type type, float scale) : this(x, y, text, type)
        {
            this.scale = scale;
            CreateText();
        }

        public Text(float x, float y, string text, Sprite.Type type, int maximumCharacterCount, float textSpeed) : this(x, y, text, type)
        {
            this.maximumCharacterCount = maximumCharacterCount;
            showAll = textSpeed <= 0 ? true : false;
            CreateText();
        }

        public Text(float x, float y, string text, Sprite.Type type, int maximumCharacterCount, float textSpeed, float scale) : this(x, y, text, type, maximumCharacterCount, textSpeed)
        {
            this.scale = scale;
            CreateText();
        }

        private void TextSetup()
        {
            switch (textType)
            {
                case Sprite.Type.Text8x8:
                    textWidth = 7;
                    spacing = 2;
                    break;
            }

            textWidth *= scale;
            spacing *= scale;

        }

        public void SetCompact(bool compact)
        {
            this.compact = compact;
            CreateText();
        }

        public void SetScale(float scale)
        {
            this.scale = scale;
            CreateText();
        }

        public void SetText(string text)
        {
            Dialogue = text;
            BreakUpWords();
            CreateText();
        }

        private void BreakUpWords()
        {
            words.Clear();
            string[] wordsArray = Regex.Split(Dialogue, "[ ]+");
            foreach (string s in wordsArray)
            {
                words.Add(s);
            }
        }

        private void CreateText()
        {
            sprites.Clear();
            TextSetup();

            int dialougeIndex = 0;
            int lineIndex = 0;
            float wordLength = 0;
            int y = 0;

            foreach (string s in words)
            {
                if (s.Length + lineIndex + 1 > maximumCharacterCount)
                {
                    wordLength = 0;
                    lineIndex = 1;
                    y++;
                }

                for (int i = 0; i < s.Length; i++)
                {
                    sprites.Add(new Sprite((int)Location.X + wordLength, (int)Location.Y + ((textWidth + spacing) * y), textType));
                    sprites.Last().SetFrame(Dialogue.Substring(dialougeIndex, 1).ToCharArray()[0], 16);
                    sprites.Last().SetScale(scale);
                    if (!showAll) { sprites.Last().Show = false; }

                    if (compact)
                    {
                        if (s[i] == 'I' || s[i] == 'i' || s[i] == '!' || s[i] == 'l' || s[i] == '.' || s[i] == ',' || s[i] == '\'' || s[i] == ':' || s[i] == ';')
                        {
                            switch (textType)
                            {
                                case Sprite.Type.Text8x8:
                                    wordLength += 3 * scale;
                                    break;
                            }
                        }
                        else if (s[i] == 't')
                        {
                            switch (textType)
                            {
                                case Sprite.Type.Text8x8:
                                    wordLength += 5 * scale;
                                    break;
                            }
                        }
                        else if (s[i] == 'f')
                        {
                            switch (textType)
                            {
                                case Sprite.Type.Text8x8:
                                    wordLength += 6 * scale;
                                    break;
                            }
                        }
                        else
                        {
                            wordLength += textWidth;
                        }
                    }
                    else
                    {
                        if (s[i] == 'I')
                        {
                            sprites.Last().SetLocation(sprites.Last().X + 4 * scale, sprites.Last().Y);
                        }
                        wordLength += textWidth;
                    }

                    dialougeIndex++;
                    lineIndex++;
                }

                // Acounts for space between words.
                dialougeIndex++;
                lineIndex++;
                wordLength += textWidth;
            }

            sprites.Reverse();
        }

        public void Update(GameTime gameTimer)
        {
            timer.Update(gameTimer);

            if (timer.Done && !showAll)
            {
                for (int i = 0; i < sprites.Count; i++)
                {
                    if (!sprites[i].Show)
                    {
                        sprites[i].Show = true;
                        break;
                    }
                    if (i == sprites.Count - 1)
                    {
                        showAll = true;
                    }
                }
                timer.Reset();
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
