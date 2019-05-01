using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Android.App;
using Android.Content;
using Android.Graphics;
using Android.OS;
using Android.Runtime;
using Android.Views;
using Android.Widget;

namespace TFLDemo.TflCommon
{
    /** An immutable result returned by a Classifier describing what was recognized. */
    public class Recognition
    {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private readonly string id;

        /** Display name for the recognition. */
        private readonly string title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private readonly float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        public Recognition(string id, string title, float confidence, RectF location)
        {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public string getId()
        {
            return id;
        }

        public string getTitle()
        {
            return title;
        }

        public float getConfidence()
        {
            return confidence;
        }

        public RectF getLocation()
        {
            return new RectF(location);
        }

        public void setLocation(RectF location)
        {
            this.location = location;
        }


        public override string ToString()
        {
            String resultString = "";
            if (id != null)
            {
                resultString += "[" + id + "] ";
            }

            if (title != null)
            {
                resultString += title + " ";
            }

            if (confidence != null)
            {
                resultString += string.Format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null)
            {
                resultString += location + " ";
            }

            return resultString.Trim();
        }
    }
}