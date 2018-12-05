//----------------------------------------------------------------------------
//  Copyright (C) 2004-2018 by EMGU Corporation. All rights reserved.       
//----------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;

namespace FaceDetection
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
            Run();
        }


        private static void Run()
        {
            IImage image;

            //Read the files as an 8-bit Bgr image  

            image = new UMat("lena.jpg", ImreadModes.Color); //UMat version
            //image = new Mat("lena.jpg", ImreadModes.Color); //CPU version

            long detectionTime;
            var faces = new List<Rectangle>();
            var eyes = new List<Rectangle>();

            DetectFace.Detect(
                image, "haarcascade_frontalface_default.xml", "haarcascade_eye.xml",
                faces, eyes,
                out detectionTime);

            foreach (var face in faces)
                CvInvoke.Rectangle(image, face, new Bgr(Color.Red).MCvScalar, 2);
            foreach (var eye in eyes)
                CvInvoke.Rectangle(image, eye, new Bgr(Color.Blue).MCvScalar, 2);

            //display the image 
            using (var iaImage = image.GetInputArray())
            {
                ImageViewer.Show(image, string.Format(
                    "Completed face and eye detection using {0} in {1} milliseconds",
                    iaImage.Kind == InputArray.Type.CudaGpuMat && CudaInvoke.HasCuda ? "CUDA"
                    : iaImage.IsUMat && CvInvoke.UseOpenCL ? "OpenCL"
                    : "CPU",
                    detectionTime));
            }
        }
    }
}