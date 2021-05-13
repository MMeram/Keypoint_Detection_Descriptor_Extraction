#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        /*normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
        preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
        BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
        description).*/
        int normType =  descriptorType.compare("DES_HOG") == 0 ? cv::NORM_L2 : cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // Is this bug still actual with the version 4.5.1
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        cv::Ptr<cv::flann::KDTreeIndexParams> indexParam = new cv::flann::KDTreeIndexParams(5);
        indexParam->setAlgorithm(cvflann::FLANN_INDEX_KDTREE);
        cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams(/*checks*/50);
        matcher = new cv::FlannBasedMatcher(indexParam,searchParams);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector<std::vector<cv::DMatch>> matchCandidates;
        float ratio = 0.80;
        matcher->knnMatch(descSource, descRef, matchCandidates, 2);
        for (auto vectElem : matchCandidates)
        {
            if (vectElem[0].distance / vectElem[1].distance < ratio)
            {
                matches.push_back(vectElem[0]);
            }
        }
        // alternative for ratio testing
        // radius match 
        //float maxDist= 0.4; 
        //std::vector<std::vector<cv::DMatch>> matches2; 
        //matcher->radiusMatch(descSource, descRef, matches2,  
        //          maxDist); // maximum acceptable distance 
                             // between the 2 descriptors 
    }

}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor = nullptr;
    if (descriptorType.compare(KPT_DESCRIPTORS[DES_INDEX::DES_BRISK]) == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare(KPT_DESCRIPTORS[DES_INDEX::DES_ORB]) == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare(KPT_DESCRIPTORS[DES_INDEX::DES_SIFT]) == 0)
    {
        extractor = cv::SIFT::create();
    }
    else if (descriptorType.compare(KPT_DESCRIPTORS[DES_INDEX::DES_AKAZE]) == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare(KPT_DESCRIPTORS[DES_INDEX::DES_FREAK]) == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else
    {
        // BRIEF
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }

    // perform feature description
    extractor->compute(img, keypoints, descriptors);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }


    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                // find the first element if kptOverlap > maxOverlap and newKeyPoint.response > kp.response
                auto it = std::find_if(keypoints.begin(), keypoints.end(),[&](cv::KeyPoint & kp){
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, kp);
                    bOverlap = kptOverlap > maxOverlap;
                    return  (bOverlap && newKeyPoint.response > kp.response);
                    });
                if (it != keypoints.end())
                {
                    *it = newKeyPoint;
                }
   
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    // visualize results
    if (bVis)
    {
        std::string windowName = "Harris NMS Corner Detection Result";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // BRISK detector / descriptor
    cv::Ptr<cv::FeatureDetector> detector = nullptr;
    if (detectorType.compare(KPT_DETECTORS[DET_INDEX::FAST]) == 0)
    {
        int intensityThreshold = 35;
        detector = cv::FastFeatureDetector::create(intensityThreshold); 
    }
    else if(detectorType.compare(KPT_DETECTORS[DET_INDEX::BRISK]) == 0)
    {
        detector = cv::BRISK::create();
    }
    else if(detectorType.compare(KPT_DETECTORS[DET_INDEX::ORB]) == 0)
    {
        detector = cv::ORB::create();
    }
    else if(detectorType.compare(KPT_DETECTORS[DET_INDEX::SIFT]) == 0)
    {
        detector = cv::SIFT::create();   
    }
    else{
        //AKAZE
        detector = cv::AKAZE::create();
    }

    detector->detect(img, keypoints);
    
     if (bVis)
    {
        std::string windowName = detectorType + " Detection Result";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
   
}