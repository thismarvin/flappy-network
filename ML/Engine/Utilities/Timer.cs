
using System;

using Microsoft.Xna.Framework;

namespace ML.Engine.Utilities
{
    class Timer
    {
        public bool Done { get; private set; }
        public bool Stop { get; set; }
        float duration;
        float elapsedTime;
        float startingTime;
        bool executeOnce;

        public Timer(float timerLength)
        {
            duration = timerLength;
        }

        public void SetDuration(float duration)
        {
            this.duration = duration;
        }

        public void Reset()
        {
            elapsedTime = 0;
            Done = false;
            executeOnce = false;
        }

        private void Setup(GameTime gameTime)
        {
            if (!executeOnce)
            {
                startingTime = (float)gameTime.TotalGameTime.TotalMilliseconds;
                executeOnce = true;
            }
        }

        public void Update(GameTime gameTime)
        {
            if (!Done && !Stop)
            {
                Setup(gameTime);
                elapsedTime = (float)gameTime.TotalGameTime.TotalMilliseconds - startingTime;

                if (elapsedTime >= duration)
                {
                    Done = true;
                }
            }
            if (Stop)
            {
                Console.WriteLine(elapsedTime);
                Console.WriteLine();
            }
        }
    }
}
