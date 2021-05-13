#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

static const std::array<std::string,7> KPT_DETECTORS{"SHITOMASI","HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};

// HARRIS, FAST, BRISK, ORB, AKAZE, SIFT, SHITOMASI
enum DET_INDEX
{
    SHITOMASI,
    HARRIS,
    FAST,
    BRISK,
    ORB,
    AKAZE,
    SIFT
};

static const std::array<std::string,6> KPT_DESCRIPTORS{"BRISK", "BRIEF","ORB", "AKAZE", "FREAK","SIFT"};
// BRIEF, ORB, FREAK, AKAZE, SIFT
enum DES_INDEX
{
    DES_BRISK,
    DES_BRIEF,
    DES_ORB,
    DES_AKAZE,
    DES_FREAK,
    DES_SIFT
};

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis=false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

#endif /* matching2D_hpp */
