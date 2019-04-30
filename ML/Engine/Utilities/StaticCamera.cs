using System;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

using ML.Engine.GameComponents;
using ML.Engine.Entities.Geometry;

namespace ML.Engine.Utilities
{
    static class StaticCamera
    {
        public static Matrix Transform { get; private set; }
        public static float Zoom { get; private set; }
        public static float VerticalLetterBox { get; private set; }
        public static float HorizontalLetterBox { get; private set; }

        private static Shape topLetterBox;
        private static Shape bottomLetterBox;
        private static Shape leftLetterBox;
        private static Shape rightLetterBox;

        public static void Initialize()
        {
            switch (Game1.GameOrientation)
            {
                case Game1.Orientation.Landscape:
                    Reset(ScreenManager.DefaultWindowWidth, ScreenManager.DefaultWindowHeight);
                    break;
                case Game1.Orientation.Portrait:
                    Reset(ScreenManager.DefaultWindowWidth, ScreenManager.DefaultWindowHeight);
                    break;
            }
        }

        private static void ResetZoom()
        {
            Zoom = Camera.Zoom;
        }

        private static void SetupLandscapeLetterBox(int windowWidth, int windowHeight)
        {
            VerticalLetterBox = (windowHeight / Zoom - Camera.ScreenBounds.Height * Camera.Scale) / 2;
            topLetterBox = new Shape(-128, -(int)VerticalLetterBox - 128, Camera.ScreenBounds.Width + 128 * 2, (int)VerticalLetterBox + 128, Color.Black);
            bottomLetterBox = new Shape(-128, Camera.ScreenBounds.Height, Camera.ScreenBounds.Width + 128 * 2, (int)VerticalLetterBox + 128, Color.Black);

            if (!ScreenManager.WideScreenSupport)
            {
                HorizontalLetterBox = (windowWidth / Zoom - Camera.ScreenBounds.Width * Camera.Scale) / 2;
                leftLetterBox = new Shape(-128 - (int)HorizontalLetterBox, -128, 128 + (int)HorizontalLetterBox, Camera.ScreenBounds.Height + 128 * 2, Color.Black);
                rightLetterBox = new Shape(Camera.ScreenBounds.Width, -128, (int)HorizontalLetterBox + 128, Camera.ScreenBounds.Height + 128 * 2, Color.Black);
            }
        }

        private static void SetupPortraitLetterBox(int windowWidth)
        {
            VerticalLetterBox = (windowWidth / Zoom - Camera.ScreenBounds.Width * Camera.Scale) / 2;
            topLetterBox = new Shape(-(int)VerticalLetterBox - 128, -128, (int)VerticalLetterBox + 128, Camera.ScreenBounds.Height + 128 * 2, Color.Black);
            bottomLetterBox = new Shape(Camera.ScreenBounds.Width, -128, (int)VerticalLetterBox + 128, Camera.ScreenBounds.Height + 128 * 2, Color.Black);
        }

        private static void FinalizeLanscapeMatrix()
        {
            // Fixed on Top Left.
            Transform = Matrix.CreateTranslation(new Vector3(HorizontalLetterBox, VerticalLetterBox, 0)) *
                        Matrix.CreateScale(new Vector3(Zoom, Zoom, 0));
        }

        private static void FinalizePortraitMatrix()
        {
            // Fixed on Top Left.
            Transform = Matrix.CreateTranslation(new Vector3(VerticalLetterBox, HorizontalLetterBox, 0)) *
                        Matrix.CreateScale(new Vector3(Zoom, Zoom, 0));
        }

        public static void Reset(int windowWidth, int windowHeight)
        {
            ResetZoom();
            switch (Game1.GameOrientation)
            {
                case Game1.Orientation.Landscape:
                    SetupLandscapeLetterBox(windowWidth, windowHeight);
                    FinalizeLanscapeMatrix();
                    break;
                case Game1.Orientation.Portrait:
                    SetupPortraitLetterBox(windowWidth);
                    FinalizePortraitMatrix();
                    break;
            }
        }

        public static void Draw(SpriteBatch spriteBatch)
        {
            spriteBatch.Begin(SpriteSortMode.Deferred, BlendState.NonPremultiplied, SamplerState.PointClamp, null, null, null, Transform);
            {
                topLetterBox.Draw(spriteBatch);
                bottomLetterBox.Draw(spriteBatch);

                if (!ScreenManager.WideScreenSupport && Game1.GameOrientation == Game1.Orientation.Landscape)
                {
                    leftLetterBox.Draw(spriteBatch);
                    rightLetterBox.Draw(spriteBatch);
                }
            }
            spriteBatch.End();
        }
    }
}