using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;

namespace PedestrianDetection
{
    internal static class Program
    {
        /// <summary>
        ///     The main entry point for the application.
        /// </summary>
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            using (var image = new Mat("C:\\Emgu\\emgucv-windesktop 3.4.3.3016\\Emgu.CV.Example\\PedestrianDetection\\pedestrian1.png"))
            {
                long processingTime;
                Rectangle[] results;

                if (CudaInvoke.HasCuda)
                    using (var gpuMat = new GpuMat(image))
                    {
                        results = FindPedestrian.Find(gpuMat, out processingTime);
                    }
                else
                    using (var uImage = image.GetUMat(AccessType.ReadWrite))
                    {
                        results = FindPedestrian.Find(uImage, out processingTime);
                    }

                foreach (var rect in results) CvInvoke.Rectangle(image, rect, new Bgr(Color.Red).MCvScalar);
                ImageViewer.Show(
                    image,
                    string.Format("Pedestrian detection using {0} in {1} milliseconds.",
                        CudaInvoke.HasCuda ? "GPU" :
                        CvInvoke.UseOpenCL ? "OpenCL" :
                        "CPU",
                        processingTime));
            }
        }
    }
}