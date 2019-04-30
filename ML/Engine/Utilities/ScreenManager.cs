
using System;
using System.Collections.Generic;
using System.Linq;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

using ML.Engine.GameComponents;
using ML.Engine.Level;

namespace ML.Engine.Utilities
{
    static class ScreenManager
    {
        public static int DisplayWidth { get; private set; }
        public static int DisplayHeight { get; private set; }
        public static int DefaultWindowHeight { get; private set; }
        public static int DefaultWindowWidth { get; private set; }
        public static float FPS { get; private set; }
        static Queue<float> sampleFPS;
        static Keys mappedKey;
        static bool released;
        static bool fullscreen;

        public static bool WideScreenSupport { get; private set; }

        public static void Initialize(int defaultWindowWidth, int defaultWindowHeight)
        {
            DisplayWidth = GraphicsAdapter.DefaultAdapter.CurrentDisplayMode.Width;
            DisplayHeight = GraphicsAdapter.DefaultAdapter.CurrentDisplayMode.Height;
            DefaultWindowWidth = defaultWindowWidth;
            DefaultWindowHeight = defaultWindowHeight;

            mappedKey = Keys.F10;
            WideScreenSupport = true;

            sampleFPS = new Queue<float>();
        }

        public static void StartFullScreen(GraphicsDeviceManager graphics)
        {
            ActivateFullScreenMode(graphics);
            fullscreen = true;
            released = false;
        }

        public static void Update(GameTime gameTime, GraphicsDeviceManager graphics)
        {
            UpdateFPS(gameTime);
            KeyboardInput(graphics);
        }

        private static void UpdateFPS(GameTime gameTime)
        {
            if ((float)gameTime.ElapsedGameTime.TotalSeconds != 0)
            {
                sampleFPS.Enqueue(1 / (float)gameTime.ElapsedGameTime.TotalSeconds);
            }

            if (sampleFPS.Count == 100)
            {
                FPS = sampleFPS.Average(i => i);
                sampleFPS.Dequeue();
            }
        }

        private static void KeyboardInput(GraphicsDeviceManager graphics)
        {
            if (Keyboard.GetState().IsKeyUp(mappedKey))
            {
                released = true;
            }

            if (Keyboard.GetState().IsKeyDown(mappedKey) && released)
            {
                if (fullscreen) { ResetScreenToDefaultDimensions(graphics); }
                else { ActivateFullScreenMode(graphics); }
                fullscreen = !fullscreen;
                released = false;
            }
        }

        private static void ActivateFullScreenMode(GraphicsDeviceManager graphics)
        {
            Camera.Reset(DisplayWidth, DisplayHeight);
            StaticCamera.Reset(DisplayWidth, DisplayHeight);
            graphics.PreferredBackBufferHeight = DisplayHeight;
            graphics.PreferredBackBufferWidth = DisplayWidth;
            graphics.ToggleFullScreen();
            graphics.ApplyChanges();
        }

        private static void ResetScreenToDefaultDimensions(GraphicsDeviceManager graphics)
        {
            Camera.Reset(DefaultWindowWidth, DefaultWindowHeight);
            StaticCamera.Reset(DefaultWindowWidth, DefaultWindowHeight);
            graphics.PreferredBackBufferHeight = DefaultWindowHeight;
            graphics.PreferredBackBufferWidth = DefaultWindowWidth;
            graphics.ToggleFullScreen();
            graphics.ApplyChanges();
        }
    }
}
