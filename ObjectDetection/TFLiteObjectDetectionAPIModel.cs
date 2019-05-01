using Android.Content.Res;
using Android.Graphics;
using Android.OS;
using Java.IO;
using Java.Nio;
using Java.Nio.Channels;
using System.Collections.Generic;
using TFLDemo.TflCommon;
using Xamarin.TensorFlow.Lite;

namespace TFLDemo.ObjectDetection
{
    public class TFLiteObjectDetectionAPIModel : IClassifier
    {
        private static readonly object /* Logger*/ LOGGER = null;//new Logger();

        // Only return this many results.
        private static readonly int NUM_DETECTIONS = 10;
        // Float model
        private static readonly float IMAGE_MEAN = 128.0f;
        private static readonly float IMAGE_STD = 128.0f;
        // Number of threads in the java app
        private static readonly int NUM_THREADS = 4;
        private bool isModelQuantized;
        // Config values.
        private int inputSize;
        // Pre-allocated buffers.
        private List<string> labels = new List<string>();
        private int[] intValues;
        // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
        // contains the location of detected boxes
        private float[][][] outputLocations;
        // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
        // contains the classes of detected boxes
        private float[][] outputClasses;
        // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
        // contains the scores of detected boxes
        private float[][] outputScores;
        // numDetections: array of shape [Batchsize]
        // contains the number of detected boxes
        private float[] numDetections;

        private ByteBuffer imgData;

        private Interpreter tfLite;

        private TFLiteObjectDetectionAPIModel() { }

        /** Memory-map the model file in Assets. */
        private static MappedByteBuffer loadModelFile(AssetManager context, string modelFilename)
        {
            try
            {
                var fd = context.OpenFd(modelFilename);
                var inputStream = new FileInputStream(fd.FileDescriptor);
                var fileChannel = inputStream.Channel;
                var mapped = fileChannel.Map(FileChannel.MapMode.ReadOnly, fd.StartOffset, fd.DeclaredLength);
                return mapped;
            }
            catch (System.Exception e)
            {
                return null;
            }
        }

        /**
         * Initializes a native TensorFlow session for classifying images.
         *
         * @param assetManager The asset manager to be used to load assets.
         * @param modelFilename The filepath of the model GraphDef protocol buffer.
         * @param labelFilename The filepath of label file for classes.
         * @param inputSize The size of image input
         * @param isQuantized Boolean representing model is quantized or not
         */
        public static IClassifier create(AssetManager assetManager, string modelFilename, string labelFilename, int inputSize, bool isQuantized)
        {
            TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

            var actualFilename = labelFilename.Split("file:///android_asset/")[1];
            var labelsInput = assetManager.Open(actualFilename);
            BufferedReader br = null;
            br = new BufferedReader(new InputStreamReader(labelsInput));
            string line;
            while ((line = br.ReadLine()) != null)
            {
                d.labels.Add(line);
            }
            br.Close();

            d.inputSize = inputSize;

            try
            {
                d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
            }
            catch (System.Exception e)
            {
                throw e;
            }

            d.isModelQuantized = isQuantized;
            // Pre-allocate buffers.
            int numBytesPerChannel;
            if (isQuantized)
            {
                numBytesPerChannel = 1; // Quantized
            }
            else
            {
                numBytesPerChannel = 4; // Floating point
            }
            d.imgData = ByteBuffer.AllocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
            d.imgData.Order(ByteOrder.NativeOrder());
            d.intValues = new int[d.inputSize * d.inputSize];

            d.tfLite.SetNumThreads(NUM_THREADS);
            d.outputLocations = CreateJagged(1, NUM_DETECTIONS, 4);
            d.outputClasses =  CreateJagged(1, NUM_DETECTIONS);
            d.outputScores =  CreateJagged(1, NUM_DETECTIONS);
            d.numDetections = new float[1];
            return d;
        }

        private static float[][][] CreateJagged(int lay1, int lay2, int lay3)
        {
            var arr = new float[lay1][][];

            for (int i = 0; i < lay1; i++)
            {
                arr[i] = CreateJagged(lay2, lay3);
            }
            return null;
        }

        private static float[][] CreateJagged(int lay1, int lay2)
        {
            var arr = new float[lay1][];

            for (int i = 0; i < lay1; i++)
            {
                arr[i] = new float[lay2];
            }

            return arr;
        }

        //override
        public List<Recognition> recognizeImage(Bitmap bitmap)
        {
            // Log this method so that it can be analyzed with systrace.
            Trace.BeginSection("recognizeImage");

            Trace.BeginSection("preprocessBitmap");
            // Preprocess the image data from 0-255 int to normalized float based
            // on the provided parameters.
            bitmap.GetPixels(intValues, 0, bitmap.Width, 0, 0, bitmap.Width, bitmap.Height);

            imgData.Rewind();
            for (int i = 0; i < inputSize; ++i)
            {
                for (int j = 0; j < inputSize; ++j)
                {
                    int pixelValue = intValues[i * inputSize + j];
                    if (isModelQuantized)
                    {
                        // Quantized model
                        imgData.Put((sbyte)((pixelValue >> 16) & 0xFF));
                        imgData.Put((sbyte)((pixelValue >> 8) & 0xFF));
                        imgData.Put((sbyte)(pixelValue & 0xFF));
                    }
                    else
                    { // Float model
                        imgData.PutFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                        imgData.PutFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                        imgData.PutFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    }
                }
            }
            Trace.EndSection(); // preprocessBitmap

            // Copy the input data into TensorFlow.
            Trace.BeginSection("feed");
            outputLocations = CreateJagged(1, NUM_DETECTIONS, 4);
            outputClasses = CreateJagged(1, NUM_DETECTIONS);
            outputScores = CreateJagged(1, NUM_DETECTIONS);
            numDetections = new float[1];


            Java.Lang.Object[] inputArray = { imgData };

            var outputMap = new Dictionary<Java.Lang.Integer, Java.Lang.Object>();
            outputMap.Add(new Java.Lang.Integer(0), outputLocations);
            outputMap.Add(new Java.Lang.Integer(1), outputClasses);
            outputMap.Add(new Java.Lang.Integer(2), outputScores);
            outputMap.Add(new Java.Lang.Integer(3), numDetections);
            Trace.EndSection();

            // Run the inference call.
            Trace.BeginSection("run");
            tfLite.RunForMultipleInputsOutputs(inputArray, outputMap);
            Trace.EndSection();

            // Show the best detections.
            // after scaling them back to the input size.
            var recognitions = new List<Recognition>(NUM_DETECTIONS);
            for (int i = 0; i < NUM_DETECTIONS; ++i)
            {
                var detection =
                    new RectF(
                        outputLocations[0][i][1] * inputSize,
                        outputLocations[0][i][0] * inputSize,
                        outputLocations[0][i][3] * inputSize,
                        outputLocations[0][i][2] * inputSize);
                // SSD Mobilenet V1 Model assumes class 0 is background class
                // in label file and class labels start from 1 to number_of_classes+1,
                // while outputClasses correspond to class index from 0 to number_of_classes

                int labelOffset = 1;
                recognitions.Add(
                    new Recognition(
                        "" + i,
                        labels[(int)outputClasses[0][i] + labelOffset],
                        outputScores[0][i],
                        detection));
            }
            Trace.EndSection(); // "recognizeImage"
            return recognitions;
        }

        public void enableStatLogging(bool logStats) { }

        public string getStatString()
        {
            return "";
        }

        public void close() { }

        public void setNumThreads(int num_threads)
        {
            if (tfLite != null) tfLite.SetNumThreads(num_threads);
        }

        public void setUseNNAPI(bool isChecked)
        {
            if (tfLite != null) tfLite.SetUseNNAPI(isChecked);
        }
    }
}