#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <list>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    std::string imgName;
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

template <typename T>
struct RingBuffer
{
    typedef T ValueType;
    RingBuffer() = delete; // default constructor is not available
    // explicit constructor
    explicit RingBuffer(const size_t rbsize)
    // one element more to delete the discarded one
        : _ringbuffer{}, _size(rbsize + 1)
    {
    }
    /*
    * Pushes to the front and removes from the 
    * end if the size exceeds ring buffer size
    * with constant complexity O(1)
    */
    void push_front(const ValueType &elem)
    {
        _ringbuffer.emplace_front(elem);
        if (_ringbuffer.size() == _size)
        {
            _ringbuffer.pop_back();
            //std::cout << "Last element discarded\n";
        }
    }
    /*
    * Returns the last, last-1, ... last-nth element
    * if the index exceeds the limit, returns the element at 0th index
    */
    ValueType& back(size_t const index=0)
    {
        return index >= _ringbuffer.size() ? *(_ringbuffer.begin()) : *std::prev(_ringbuffer.end(), index+1);
    }

    /*
    * Returns the first, first+1, ... first+nth element
    * if the index exceeds the limit, returns the last element
    */
    ValueType& front(size_t const index=0)
    {
        return index >= _ringbuffer.size() ? *(_ringbuffer.end()) : *std::next(_ringbuffer.begin(), index);
    }

    size_t size()
    {
        return _ringbuffer.size();
    }
    // left public for accessing public methods of vector
    

private:
    std::list<ValueType> _ringbuffer;
    size_t _size;
};

#endif /* dataStructures_h */
