using Android.Graphics;
using System;
using System.Collections.Generic;
using TFLDemo.TflCommon;

namespace TFLDemo.ObjectDetection
{
    public interface IClassifier
    {
        List<Recognition> recognizeImage(Bitmap bitmap);

        void enableStatLogging(bool debug);

        String getStatString();

        void close();

        void setNumThreads(int num_threads);

        void setUseNNAPI(bool isChecked);
    }

}