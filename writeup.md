[//]: # (Image References)
[image1]: ./output_images/hog_extractor_car.jpg
[image2]: ./output_images/hog_extractor_car_luv_channel1.jpg
[image3]: ./output_images/hog_extractor_car_luv_channel2.jpg
[image4]: ./output_images/hog_extractor_car_luv_channel3.jpg
[image5]: ./output_images/hog_extractor_video.jpg
[image6]: ./output_images/hog_extractor_video_cropped.jpg
[image7]: ./output_images/hog_extractor_video_luv_channel1.jpg
[image8]: ./output_images/hog_extractor_video_luv_channel2.jpg
[image9]: ./output_images/hog_extractor_video_luv_channel3.jpg
[image10]: ./output_images/predict_windows_multiple_scales.jpg
[image11]: ./output_images/predict_windows_heatmap.jpg
[image12]: ./output_images/predict_windows_bounding_boxes.jpg
[image13]: ./output_images/frame1_sliding_window.jpg
[image14]: ./output_images/frame1_heatmap.jpg
[image15]: ./output_images/frame1_bounding_boxes.jpg
[image16]: ./output_images/frame2_sliding_window.jpg
[image17]: ./output_images/frame2_heatmap.jpg
[image18]: ./output_images/frame2_bounding_boxes.jpg
[image19]: ./output_images/frame3_sliding_window.jpg
[image20]: ./output_images/frame3_heatmap.jpg
[image21]: ./output_images/frame3_bounding_boxes.jpg
[image22]: ./output_images/frame4_sliding_window.jpg
[image23]: ./output_images/frame4_heatmap.jpg
[image24]: ./output_images/frame4_bounding_boxes.jpg
[image25]: ./output_images/frame5_sliding_window.jpg
[image26]: ./output_images/frame5_heatmap.jpg
[image27]: ./output_images/frame5_bounding_boxes.jpg
[image28]: ./output_images/frame6_sliding_window.jpg
[image29]: ./output_images/frame6_heatmap.jpg
[image30]: ./output_images/frame6_bounding_boxes.jpg


## Code Location
All code mentioned below is located at the [Jupyter notebook](./notebook.ipynb), and it's broken down by different sections (like `1 HOG Feature Extraction`) and subsections (like `1.1 Extractor Class`).

## Histogram of Oriented Gradients (HOG)

### Explain how (and identify where in your code) you extracted HOG features from the training images.

Corresponding code is located at the `1 HOG Feature Extraction`.

Knowing that later we need to do HOG features subsampling, i.e. extracting HOG features on entire image first, and do subsampling to get HOG features on a part of an image, I started by implementing a class `HogExtractor` that encapsulate this kind of logic. Each instance of this class wraps an image (which could be an entire camera image or a small 64x64 image), and at initialization time the HOG features on the entire image is extracted with certain parameters. Then, this class mainly provides 2 functions: (1) `features()` returns the HOG features for the whole image and `window_features(window)` returns the HOG features within a given window. Another benefit of using this class is that it's easy to ensure we use the same parameters for training/test data and the video frames.

Below are an image from given dataset and its HOG images using LUV color space:

![Car Image][image1] ![Car Image (L channel)][image2] ![Car Image (U channel)][image3] ![Car Image (V channel)][image4]

Below are an image from one of the test images and its HOG images using LUV color space:

![Video Image][image5] ![Video Image (Cropped)][image6] ![Video Image (L channel)][image7] ![Video Image (U channel)][image8] ![Video Image (V channel)][image9]

### Explain how you settled on your final choice of HOG parameters.

Like the course material, I'm choosing `pixels_per_cell = 8`, `cells_per_block = 2` and `orient = 2`. I was not sure what color space to use, so I tried different color spaces and trying to find out what results in a better test accuracy using SVM classifier. Below is the training/testing accuracy by using different color spaces. From the results I decided to use `LUV`.

| Color Space   | Training Accuracy | Testing Accuracy  |
|:--------------|------------------:| -----------------:|
| RGB           | 0.9991            | 0.9916            |
| HSV           | 0.9996            | 0.9927            |
| LUV           | 0.9995            | 0.9947            |
| YCrCb         | 0.9996            | 0.9938            |

### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all default options: `auto` gamma, `rbf` kernal and linear classifier. Code can be found at `3 Training SVM`.

## Sliding Window Search

### Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search in 6 steps (corresponding to subsections 5.1 - 5.4 in my code):
1. Given an image, a window size and the overlap distance, generate all windows
2. Given an image and a window, predict whether the window is a car image
3. Given an image, a window and a scale, scale down the window and predict whether the window on the scaled-down image contains a car
4. Putting everything together: given a window and a list of scales, for each scale, generate all windows based on the scaled-down image, predict whether each window is a car image. If there is a match, scale the window up to match the original size of the image, and collect the positive windows into a list.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are 3 images demonstrating how the pipeline is working:
![Sliding Window Search with Multiple Scales][image10] ![Heatmap][image11] ![Bounding Boxes from Heatmap][image12]

I cropped the image before processing (400:656 on y-axis) because the upper part of the image is unlikely to have cars.

## Video Implementation

### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)

### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After I can identify windows which look like a car, I did 2 steps to draw the bounding boxes on separate car images (corresponding to 6.1 - 6.2 in my code):
1. For all positive windows in a frame, increase the 'heat' on the window in a heatmap
2. For the heatmap, call `scipy.ndimage.measurements.label()` to label the different separate 'cars', and finally get the bounding boxes on those labels.

#### Here are six frames with detected windows:

![Frame 1][image13] ![Frame 2][image16] ![Frame 3][image19] ![Frame 4][image22] ![Frame 5][image25] ![Frame 6][image28]

#### Here are heatmaps from those windows
![Frame 1][image14] ![Frame 2][image17] ![Frame 3][image20] ![Frame 4][image23] ![Frame 5][image26] ![Frame 6][image29]

### Here the resulting bounding boxes:
![Frame 1][image15] ![Frame 2][image18] ![Frame 3][image21] ![Frame 4][image24] ![Frame 5][image27] ![Frame 6][image30]

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The bounding boxes are not very stable from frame to frame, possibly because the sliding window search is not granular enough so that when a vehicle moves a small distance in the next frame the window can also only moves a small distance. I could probably improve this by identifying the same vehicles between frames and do some estimation on the size and the position of the vehicle, based on 2 assumptions: the size will not change a lot and the speed will not change a lot.
2. When 2 vehicles overlap, they are identified as one car. Not sure how to improve that though.
