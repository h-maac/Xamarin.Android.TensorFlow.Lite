using Android.App;
using Android.Graphics;
using Android.OS;
using Java.IO;
using Java.Nio;
using System;
using System.Collections.Generic;
using TFLDemo.TflCommon;
using Xamarin.TensorFlow.Lite;

namespace TFLDemo.Classifier
{
    /** The model type used for classification. */
    public enum Model
    {
        FLOAT,
        QUANTIZED,
    }

    public enum Device
    {
        CPU,
        NNAPI,
        GPU
    }

    public abstract class ClassifierBase
    {
        private static readonly object /*Logger */ LOGGER = null; // new Logger();

        /** Number of results to show in the UI. */
        private static readonly int MAX_RESULTS = 3;

        /** Dimensions of inputs. */
        private static readonly int DIM_BATCH_SIZE = 1;

        private static readonly int DIM_PIXEL_SIZE = 3;

        /** Preallocated buffers for storing image data in. */
        private readonly int[] intValues;

        /** Options for configuring the Interpreter. */
        private readonly Interpreter.Options tfliteOptions = new Interpreter.Options();

        /** The loaded TensorFlow Lite model. */
        private MappedByteBuffer tfliteModel;

        /** Labels corresponding to the output of the vision model. */
        private List<string> labels;

        /** Optional GPU delegate for accleration. */
        private object /*GpuDelegate */ gpuDelegate = null;

        /** An instance of the driver class to run model inference with Tensorflow Lite. */
        protected Interpreter tflite;

        /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
        protected ByteBuffer imgData = null;

        /**
         * Creates a classifier with the provided configuration.
         *
         * @param activity The current Activity.
         * @param model The model to use for classification.
         * @param device The device to use for classification.
         * @param numThreads The number of threads to use for classification.
         * @return A classifier with the desired configuration.
         */
        public static ClassifierBase create(Activity activity, Model model, Device device, int numThreads)
        {
            //if (model == Model.QUANTIZED)
            //{
            //    return new ClassifierQuantizedMobileNet(activity, device, numThreads);
            //}
            //else
            //{
            //    return new ClassifierFloatMobileNet(activity, device, numThreads);
            //}
            return null;
        }

        /** Initializes a {@code Classifier}. */
        protected ClassifierBase(Activity activity, Device device, int numThreads)
        {
            intValues = new int[getImageSizeX() * getImageSizeY()];
            tfliteModel = loadModelFile(activity);
            switch (device)
            {
                case Device.NNAPI:
                    tfliteOptions.SetUseNNAPI(true);
                    break;
                case Device.GPU:
                    //gpuDelegate = new Xamarin.TensorFlow.Lite. GpuDelegate();
                    //tfliteOptions.addDelegate(gpuDelegate);
                    break;
                case Device.CPU:
                    break;
            }
            tfliteOptions.SetNumThreads(numThreads);
            tflite = new Interpreter(tfliteModel, tfliteOptions);
            labels = loadLabelList(activity);
            imgData =
                    ByteBuffer.AllocateDirect(
                        DIM_BATCH_SIZE
                            * getImageSizeX()
                            * getImageSizeY()
                            * DIM_PIXEL_SIZE
                            * getNumBytesPerChannel());
            imgData.Order(ByteOrder.NativeOrder());
            //LOGGER.d("Created a Tensorflow Lite Image Classifier.");
        }

        /** Reads label list from Assets. */
        private List<string> loadLabelList(Activity activity)
        {
            List<string> labels = new List<string>();
            BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.Assets.Open(getLabelPath())));
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                labels.Add(line);
            }
            reader.Close();
            return labels;
        }

        /** Memory-map the model file in Assets. */
        private MappedByteBuffer loadModelFile(Activity activity)
        {
            var fileDescriptor = activity.Assets.OpenFd(getModelPath());
            var inputStream = new FileInputStream(fileDescriptor.FileDescriptor);
            var fileChannel = inputStream.Channel;
            var startOffset = fileDescriptor.StartOffset;
            var declaredLength = fileDescriptor.DeclaredLength;
            return fileChannel.Map(Java.Nio.Channels.FileChannel.MapMode.ReadOnly, startOffset, declaredLength);
        }

        /** Writes Image data into a {@code ByteBuffer}. */
        private void convertBitmapToByteBuffer(Bitmap bitmap)
        {
            if (imgData == null)
            {
                return;
            }
            imgData.Rewind();
            bitmap.GetPixels(intValues, 0, bitmap.Width, 0, 0, bitmap.Width, bitmap.Height);
            // Convert the image to floating point.
            int pixel = 0;
            long startTime = SystemClock.UptimeMillis();
            for (int i = 0; i < getImageSizeX(); ++i)
            {
                for (int j = 0; j < getImageSizeY(); ++j)
                {
                    int val = intValues[pixel++];
                    addPixelValue(val);
                }
            }
            long endTime = SystemClock.UptimeMillis();
            //LOGGER.v("Timecost to put values into ByteBuffer: " + (endTime - startTime));
        }

        /** Runs inference and returns the classification results. */
        public List<Recognition> recognizeImage(Bitmap bitmap)
        {
            // Log this method so that it can be analyzed with systrace.
            Trace.BeginSection("recognizeImage");

            Trace.BeginSection("preprocessBitmap");
            convertBitmapToByteBuffer(bitmap);
            Trace.EndSection();

            // Run the inference call.
            Trace.BeginSection("runInference");
            long startTime = SystemClock.UptimeMillis();
            runInference();
            long endTime = SystemClock.UptimeMillis();
            Trace.EndSection();
            //LOGGER.v("Timecost to run model inference: " + (endTime - startTime));

            // Find the best classifications.
            Queue<Recognition> pq = new Queue<Recognition>(3);
            //new PriorityQueue<Recognition>(
            //    3,
            //new Comparator<Recognition>()
            //{
            //  // override
            //      public int compare(Recognition lhs, Recognition rhs)
            //    {
            //        // Intentionally reversed to put high confidence at the head of the queue.
            //        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
            //    }
            //});
            for (int i = 0; i < labels.Count; ++i)
            {
                pq.Enqueue(
                    new Recognition(
                        "" + i,
                        labels.Count > i ? labels[i] : "unknown",
                        getNormalizedProbability(i),
                        null));
            }
            var recognitions = new List<Recognition>();
            int recognitionsSize = Math.Min(pq.Count, MAX_RESULTS);
            for (int i = 0; i < recognitionsSize; ++i)
            {
                recognitions.Add(pq.Dequeue());
            }
            Trace.EndSection();
            return recognitions;
        }

        /** Closes the interpreter and model to release resources. */
        public void close()
        {
            if (tflite != null)
            {
                tflite.Close();
                tflite = null;
            }
            //if (gpuDelegate != null)
            //{
            //    gpuDelegate.Close();
            //    gpuDelegate = null;
            //}
            tfliteModel = null;
        }

        /**
         * Get the image size along the x axis.
         *
         * @return
         */
        public abstract int getImageSizeX();

        /**
         * Get the image size along the y axis.
         *
         * @return
         */
        public abstract int getImageSizeY();

        /**
         * Get the name of the model file stored in Assets.
         *
         * @return
         */
        protected abstract string getModelPath();

        /**
         * Get the name of the label file stored in Assets.
         *
         * @return
         */
        protected abstract string getLabelPath();

        /**
         * Get the number of bytes that is used to store a single color channel value.
         *
         * @return
         */
        protected abstract int getNumBytesPerChannel();

        /**
         * Add pixelValue to byteBuffer.
         *
         * @param pixelValue
         */
        protected abstract void addPixelValue(int pixelValue);

        /**
         * Read the probability value for the specified label This is either the original value as it was
         * read from the net's output or the updated value after the filter was applied.
         *
         * @param labelIndex
         * @return
         */
        protected abstract float getProbability(int labelIndex);

        /**
         * Set the probability value for the specified label.
         *
         * @param labelIndex
         * @param value
         */
        protected abstract void setProbability(int labelIndex, int value);

        /**
         * Get the normalized probability value for the specified label. This is the readonly value as it
         * will be shown to the user.
         *
         * @return
         */
        protected abstract float getNormalizedProbability(int labelIndex);

        /**
         * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
         * provided by getProbability().
         *
         * <p>This additional method is necessary, because we don't have a common base for different
         * primitive data types.
         */
        protected abstract void runInference();

        /**
         * Get the total number of labels.
         *
         * @return
         */
        protected int getNumLabels()
        {
            return labels.Count;
        }
    }
}