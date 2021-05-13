# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The project consists of four parts:

1. The Data Buffer: 
   1. The Data buffer is implemented using the std::list (single linked list). Pushing elements to the front and removing them from the back has a constant
      time complexity, O(1) and therefore quite efficient

2. Keypoint Detection: 
   1. Implemented detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT using the OpenCV library, detectors can be selected from a string array via index values
   2. Removed the keypoints outside of the predefined window by employing "remove-erase" idiom und "inside" method of the Point structure

3. Descriptor Extraction & Matching: 
   1. Implemented descriptors BRIEF, ORB, FREAK, AKAZE and SIFT using the OpenCV library, descriptors can be selected from a string array via index values
   2. Implemented FLANN matching as well as k-nearest neighbor selection. Both methods can be selectable using the respective strings in the main function. Note that the norm (L1,L2 or Hamming Distance) selection depends on the descriptor type and it's also incorporated
   3. Implemented Knn-matching by selecting 2 as the best match candidates. The ratio is taken as 0.8

4. Performance Evaluation: 
   1. Results are in the spreadsheet
   2. Results are in the spreadsheet
   3. Results are in the spreadsheet

As a result the following detector/descriptor pairs are selected as the top-3:

FAST Detector - BRIEF Descriptor,  9.99 ms 
FAST Detector - ORB Descriptor,    27.50 ms
ORB Detector - BRIEF Descriptor,   51.51 ms

They are very fast for 10 consecutive images and score also excellent in terms of matched points. 
