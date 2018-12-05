//----------------------------------------------------------------------------
//  Copyright (C) 2004-2018 by EMGU Corporation. All rights reserved.       
//----------------------------------------------------------------------------

using System.Diagnostics;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
#if !(__IOS__ || NETFX_CORE)
using Emgu.CV.Cuda;

#endif

namespace PedestrianDetection
{
    public static class FindPedestrian
    {
        /// <summary>
        ///     Find the pedestrian in the image
        /// </summary>
        /// <param name="image">The image</param>
        /// <param name="processingTime">The processing time in milliseconds</param>
        /// <returns>The region where pedestrians are detected</returns>
        public static Rectangle[] Find(IInputArray image, out long processingTime)
        {
            Stopwatch watch;
            Rectangle[] regions;

            using (var iaImage = image.GetInputArray())
            {
#if !(__IOS__ || NETFX_CORE)
                //if the input array is a GpuMat
                //check if there is a compatible Cuda device to run pedestrian detection
                if (iaImage.Kind == InputArray.Type.CudaGpuMat)
                    using (var des = new CudaHOG(new Size(64, 128), new Size(16, 16), new Size(8, 8), new Size(8, 8)))
                    {
                        des.SetSVMDetector(des.GetDefaultPeopleDetector());

                        watch = Stopwatch.StartNew();
                        using (var cudaBgra = new GpuMat())
                        using (var vr = new VectorOfRect())
                        {
                            CudaInvoke.CvtColor(image, cudaBgra, ColorConversion.Bgr2Bgra);
                            des.DetectMultiScale(cudaBgra, vr);
                            regions = vr.ToArray();
                        }
                    }
                else
#endif
                    using (var des = new HOGDescriptor())
                    {
                        des.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
                        watch = Stopwatch.StartNew();

                        var results = des.DetectMultiScale(image);
                        regions = new Rectangle[results.Length];
                        for (var i = 0; i < results.Length; i++)
                            regions[i] = results[i].Rect;
                        watch.Stop();
                    }

                processingTime = watch.ElapsedMilliseconds;

                return regions;
            }
        }
    }
}