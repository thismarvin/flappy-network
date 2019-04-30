
using System;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;

using ML.Engine.Utilities;
using ML.Engine.Level;
using ML.Engine.Resources;

namespace ML.Engine.GameComponents
{
    public class Game1 : Game
    {
        GraphicsDeviceManager graphics;
        SpriteBatch spriteBatch;

        string title;
        int defaultWindowHeight;
        int defaultWindowWidth;

        public enum Mode
        { Menu, Playfield, None }
        public static Mode GameMode { get; set; }

        public enum Orientation
        { Landscape, Portrait }
        public static Orientation GameOrientation { get; set; }

        public static bool ExitGame { get; set; }
        public static bool DebugMode { get; set; }
        bool released;

        public Game1()
        {
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";

            title = "Flappy Bird meets Machine Learning";

            // Toggle Mouse Visibility.
            IsMouseVisible = true;

            int displayWidth = GraphicsAdapter.DefaultAdapter.CurrentDisplayMode.Width;
            int displayHeight = GraphicsAdapter.DefaultAdapter.CurrentDisplayMode.Height;

            GameOrientation = Orientation.Landscape;
            GameMode = Mode.Playfield;

            defaultWindowWidth = 480 * 2;
            defaultWindowHeight = 270 * 2;

            // Set Screen Dimensions.
            graphics.PreferredBackBufferWidth = defaultWindowWidth;
            graphics.PreferredBackBufferHeight = defaultWindowHeight;

            EnableVSync(true);
            //SetTargetFPS(60);

            graphics.ApplyChanges();
        }

        private void EnableVSync(bool vsync)
        {
            if (vsync)
            {
                graphics.SynchronizeWithVerticalRetrace = true;
                base.IsFixedTimeStep = false;
            }
        }

        private void SetTargetFPS(int fps)
        {
            TargetElapsedTime = TimeSpan.FromTicks((long)(1f / fps * 10000000));
        }

        private void DebugModeToggle()
        {
            if (Keyboard.GetState().IsKeyDown(Keys.F12) && released)
            {
                DebugMode = !DebugMode;
                released = false;
            }
            if (!released && Keyboard.GetState().IsKeyUp(Keys.F12)) { released = true; }
            if (DebugMode) { Window.Title = title + " " + Math.Round(ScreenManager.FPS) + " FPS"; }
            else { Window.Title = title; }
        }

        private void ExitGameLogic()
        {
            if (Keyboard.GetState().IsKeyDown(Keys.Escape)) { ExitGame = true; }
            if (ExitGame) { Exit(); }
        }

        protected override void Initialize()
        {
            ShapeManager.Initialize(graphics);
            ScreenManager.Initialize(defaultWindowWidth, defaultWindowHeight);
            Camera.Initialize();
            StaticCamera.Initialize();

            base.Initialize();
        }

        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);

            Assets.LoadContent(Content);

            Playfield.Initialize();

            // Start Game FullScreen.
            //ScreenManager.StartFullScreen(graphics);
        }

        protected override void UnloadContent()
        {
            base.UnloadContent();

            spriteBatch.Dispose();
            ShapeManager.UnloadContent();
        }

        protected override void Update(GameTime gameTime)
        {
            ExitGameLogic();
            DebugModeToggle();

            ScreenManager.Update(gameTime, graphics);

            switch (GameMode)
            {
                case Mode.Menu:
                    break;
                case Mode.Playfield:
                    Playfield.Update(gameTime);
                    break;
            }

            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Palette.MidnightBlack);

            switch (GameMode)
            {
                case Mode.Playfield:
                    Playfield.Draw(spriteBatch);                    
                    break;
            }

            StaticCamera.Draw(spriteBatch);
            base.Draw(gameTime);
        }
    }
}
