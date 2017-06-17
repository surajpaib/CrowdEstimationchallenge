# CrowdEstimationChallenge


## Problem

Given Set: 3 overhead images from a Drone Camera
Objective: Develop an algorithm to count the approximate number of people


## Brainstorming
Idea 1:  Create CNN model with labelled segmented people and train over images to recognize people-like features. Shortcomings: 3 training images, even with multiple augmentations, the total number may not exceed more than 4 digits. This will fail to give out satisfactory results
Idea 2:  Image processing pipeline + ML Model. Use image pre-processing to separate foreground from background and then use foreground features with Numerical ML Models / Regression to train for the number of people in the image.
Idea 3: Use mostly image processing algorithms but use ML to identify parameters of some of these algorithms. Eg: SimpleBlobDetector Parameters. ( Multiple ML models for each attribute)


## Approach

Idea 3 was picked as most suitable owing to less dependency on ML ( owing to lack of sample training images for the other two methods )

Possible Preprocessing:
 1. Background Subtraction from two or more images
 2. Convert to Fourier domain to remove low freq components.

 Pre-processed image can then be thresholded to obtain features to identify people.
 This image would need some morphological operations to obtain better feature resolution,
 Erosion for removing noise in the image and dilation for emphasising on the people-like features.
 Opening is used here for better feature resolution.
 Post morphological operations, we pass the image to a Simple Blob Detector.

 Parameters for the simple blob detector are set to detect people like features. This parameter selection can be done via means of an ML algorithm such as Boosting Trees. ( Not implemented due to lack of time and samples for ML training. )

 Parameters are manually set to detect blobs of a certain area, convexity, etc.

 This algorithm would help us get a rough estimate of number of people in an image


