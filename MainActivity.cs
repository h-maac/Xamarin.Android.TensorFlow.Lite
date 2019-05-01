using Android.App;
using Android.OS;
using Android.Support.V7.App;
using TFLDemo.ObjectDetection;

namespace TFLDemo
{
    [Activity(Label = "@string/app_name", Theme = "@style/AppTheme", MainLauncher = true)]
    public class MainActivity : AppCompatActivity
    {
        const int TF_OD_API_INPUT_SIZE = 300;
        const bool TF_OD_API_IS_QUANTIZED = true;
        const string TF_OD_API_MODEL_FILE = "detect.tflite";
        const string TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);
            // Set our view from the "main" layout resource
            SetContentView(Resource.Layout.activity_main);
            var detector =
                TFLiteObjectDetectionAPIModel.create(
                    Assets,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_INPUT_SIZE,
                    TF_OD_API_IS_QUANTIZED);
            var stream = Assets.Open("img/img1.jpg");
            var bitmap = Android.Graphics.BitmapFactory.DecodeStream(stream);
            detector.recognizeImage(bitmap);
        }
    }
}