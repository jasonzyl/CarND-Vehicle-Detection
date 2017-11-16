##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_extractor_car.jpg
[image2]: ./output_images/hog_extractor_car_luv_channel1.jpg
[image3]: ./output_images/hog_extractor_car_luv_channel2.jpg
[image4]: ./output_images/hog_extractor_car_luv_channel3.jpg
[image5]: ./output_images/hog_extractor_video.jpg
[image6]: ./output_images/hog_extrhog_extractor_video_cropped.jpg
[image7]: ./output_images/hog_extractor_video_luv_channel1.jpg
[image8]: ./output_images/hog_extractor_video_luv_channel2.jpg
[image9]: ./output_images/hog_extractor_video_luv_channel3.jpg
[image10]: ./output_images/predict_windows_multiple_scales.jpg
[image11]: ./output_images/predict_windows_heatmap.jpg
[image12]: ./output_images/predict_windows_bounding_boxes.jpg

[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Code Location
All code mentioned below is located at the [Jupyter notebook](./notebook.ipynb), and it's broken down by different sections (like `1 HOG Feature Extraction`) and subsections (like `1.1 Extractor Class`).

## Histogram of Oriented Gradients (HOG)

### Explain how (and identify where in your code) you extracted HOG features from the training images.

Corresponding code is located at the `1 HOG Feature Extraction`.

Knowing that later we need to do HOG features subsampling, i.e. extracting HOG features on entire image first, and do subsampling to get HOG features on a part of an image, I started by implementing a class `HogExtractor` that encapsulate this kind of logic. Each instance of this class wraps an image (which could be an entire camera image or a small 64x64 image), and at initialization time the HOG features on the entire image is extracted with certain parameters. Then, this class mainly provides 2 functions: (1) `features()` returns the HOG features for the whole image and `window_features(window)` returns the HOG features within a given window. Another benefit of using this class is that it's easy to ensure we use the same parameters for training/test data and the video frames.

Below are an image from given dataset and its HOG images using LUV color space:

![Car Image][image1] ![Car Image (L channel)][image2] ![Car Image (U channel)][image3] ![Car Image (V channel)][image4]

Below are an image from one of the test images and its HOG images using LUV color space:

![Video Image][image5]

![Video Image (Cropped)][image6] ![Video Image (L channel)][image7] ![Video Image (U channel)][image8] ![Video Image (V channel)][image9]

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

I implemented the sliding window search in 6 steps (corresponding to subsections 5.1 - 6.2 in my code):
1. Given an image, a window size and the overlap distance, generate all windows
2. Given an image and a window, predict whether the window is a car image
3. Given an image, a window and a scale, scale down the window and predict whether the window on the scaled-down image contains a car
4. Putting everything together: given a window and a list of scales, for each scale, generate all windows based on the scaled-down image, predict whether each window is a car image. If there is a match, scale the window up to match the original size of the image, and collect the positive windows into a list.
5. For all positive windows in a frame, increase the 'heat' on the window in a heatmap
6. For the heatmap, call `scipy.ndimage.measurements.label()` to label the different separate 'cars', and finally get the bounding boxes on those labels.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are 3 images demonstrating how the pipeline is working:
![Sliding Window Search with Multiple Scales][image10] ![Heatmap][image11] ![Bounding Boxes from Heatmap][image12]

I cropped the image before processing (400:656 on y-axis) because the upper part of the image is unlikely to have cars.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
