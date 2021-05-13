/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include <array>

using namespace std;

static constexpr int ONLY_DETECTORS = 1;
static constexpr int MATCH_POINTS = 2;
static constexpr int DET_DES_PERFORMANCE = 3;



void mainLoop(std::string const& detectorType, std::string const& descriptorType, const int runType, const bool bVis = false)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time 
    RingBuffer<DataFrame> dataBuffer(dataBufferSize);// list of data frames which are held in memory at the same time

    // Performance Index
    float totalMatches = 0.0;
    float totalDesDetTime = 0.0;
    int totalKeyPoints = 0;
    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
        img.release();
    

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        frame.imgName = imgFullFilename;
        dataBuffer.push_front(frame);

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        // reserve memory for 5000 kpts
        keypoints.reserve(5000);

        double t = (double)cv::getTickCount();
        if (detectorType.compare(KPT_DETECTORS[DET_INDEX::SHITOMASI]) == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, bVis);
        }
        else if (detectorType.compare(KPT_DETECTORS[DET_INDEX::HARRIS]) == 0)
        {
            detKeypointsHarris(keypoints, imgGray, bVis);
        }
        else
        {
            detKeypointsModern(keypoints,imgGray,detectorType,bVis);
        }
    
        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;

        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(),[&](auto & kpts){
                // remove-erase idiom, first move the unused entries to the beginning and then erase by the iterator position
                return !kpts.pt.inside(vehicleRect);
            }), keypoints.end());
        }
        
        // for the measurement
        if (runType == ONLY_DETECTORS || runType == DET_DES_PERFORMANCE)
        {
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            totalKeyPoints += keypoints.size();
            totalDesDetTime += 1000 * t / 1.0 ;
            if (runType == ONLY_DETECTORS)
                continue;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer.front().keypoints = keypoints;
    
        /* EXTRACT KEYPOINT DESCRIPTORS */

        t = (double)cv::getTickCount();
        cv::Mat descriptors;
        descKeypoints(dataBuffer.front().keypoints, dataBuffer.front().cameraImg, descriptors, descriptorType);
        if (runType == DET_DES_PERFORMANCE)
        {
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            totalDesDetTime += 1000 * t / 1.0 ;
        }

        // push descriptors for current frame to end of data buffer
        dataBuffer.front().descriptors = descriptors;

        // run this only for match point test
        if (dataBuffer.size() > 1 && runType == MATCH_POINTS) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            matches.reserve(5000);
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorKind = descriptorType.compare(KPT_DESCRIPTORS[DES_SIFT]) == 0 ? "DES_HOG" : "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            t = (double)cv::getTickCount();
            matchDescriptors(dataBuffer.front().keypoints, dataBuffer.front(1).keypoints,
                             dataBuffer.front().descriptors, dataBuffer.front(1).descriptors,
                             matches, descriptorKind, matcherType, selectorType);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            totalMatches += matches.size();
            // store matches in current data frame
            dataBuffer.front().kptMatches = matches;

            // visualize matches between current and previous image
            if (bVis)
            {
                cv::Mat matchImg = dataBuffer.front().cameraImg.clone();
                cv::drawMatches( dataBuffer.front().cameraImg,  dataBuffer.front().keypoints,
                                 dataBuffer.front(1).cameraImg,  dataBuffer.front(1).keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
           
        }

    } // eof loop over all images
    if (runType == ONLY_DETECTORS)
    {
        // total number of keypoints per detector
        cout << detectorType <<" Detector, "<< totalKeyPoints   << '\n';
    }
    else if (runType == DET_DES_PERFORMANCE)
    {
        // total runtime for keypoint detection and descriptor extraction
        cout <<detectorType <<" Detector - "<< descriptorType<<" Descriptor, " << totalDesDetTime << '\n';
    }
    else
    {
         // total number of matches
        cout <<detectorType <<" Detector - "<< descriptorType<<" Descriptor, " << totalMatches  << '\n';      
    }
}


/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    int runType = MATCH_POINTS;
    mainLoop(KPT_DETECTORS[ORB], KPT_DESCRIPTORS[DES_BRIEF], runType, true);
    // Best combinations
    // 1) FAST Detector - BRIEF Descriptor
    // 2)FAST Detector - ORB Descriptor
    // 3)ORB Detector - BRIEF Descriptor


    // Uncomment the section below to enable performance measurements
    /*
    switch (runType)
    {
    case ONLY_DETECTORS:
        for (auto const &detector : KPT_DETECTORS)
        {
            mainLoop(detector, "", runType);
        }
        break;
    case MATCH_POINTS:
        for (auto const &detector : KPT_DETECTORS)
        {
            if (detector.compare(KPT_DETECTORS[SIFT]) == 0)
                continue;
            for (auto const &descriptor : KPT_DESCRIPTORS)
            {
                if ((detector.compare(KPT_DETECTORS[AKAZE]) != 0) && (descriptor.compare(KPT_DESCRIPTORS[DES_AKAZE]) == 0))
                {
                    // AKAZE descriptors expect AKAZE keypoints
                    continue;
                }
                mainLoop(detector, descriptor, runType);
            }
        }
        // run this for SIFT detector type
         // SIFT Detector gives OOM error with ORB and BRISK descriptors
        mainLoop(KPT_DETECTORS[SIFT], KPT_DESCRIPTORS[DES_SIFT], runType);
        mainLoop(KPT_DETECTORS[SIFT], KPT_DESCRIPTORS[DES_BRIEF], runType);
        mainLoop(KPT_DETECTORS[SIFT], KPT_DESCRIPTORS[DES_FREAK], runType);
        break;
    case DET_DES_PERFORMANCE:
        for (auto const &detector : KPT_DETECTORS)
        {
            // SIFT requires a lot of memory
            if (detector.compare(KPT_DETECTORS[SIFT]) == 0)
                continue;
            for (auto const &descriptor : KPT_DESCRIPTORS)
            {
                if ((detector.compare(KPT_DETECTORS[AKAZE]) != 0) && (descriptor.compare(KPT_DESCRIPTORS[DES_AKAZE]) == 0))
                {
                    // AKAZE descriptors expect AKAZE keypoints
                    continue;
                }
                mainLoop(detector, descriptor, runType);
            }
        }
        // run this for SIFT detector type
        // SIFT Detector gives OOM error with ORB and BRISK descriptors
        
        mainLoop(KPT_DETECTORS[SIFT], KPT_DESCRIPTORS[DES_SIFT], runType);
        mainLoop(KPT_DETECTORS[SIFT], KPT_DESCRIPTORS[DES_BRIEF], runType);
        mainLoop(KPT_DETECTORS[SIFT], KPT_DESCRIPTORS[DES_FREAK], runType);
     
        break;
    default:
        break;
    }

    */
    return 0;
}