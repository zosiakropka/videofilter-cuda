/* 
 * File:   filter.h
 * Author: Zosia Sobocinska
 *
 * Created on December 28, 2013, 6:15 PM
 */

#ifndef FILTER_H
#define	FILTER_H

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using cv::VideoCapture;
using cv::VideoWriter;
using cv::Mat;

typedef unsigned char uchar;

class VideoFilter {
protected:

    double RATIO;

public:

    void filter(VideoCapture v_in, VideoWriter v_out);

    inline void set_ratio(double ratio = 1.0) {
        this->RATIO = ratio;
    }

    virtual void process(uchar const* bytes_in, uchar* bytes_out,
            uint cols, uint rows, uint channels, uint step_in, uint step_out) {
    };

    static inline uint get_index(uint col, uint row, uint channels, uint step) {
        return col * channels + step * row;
    }

    static inline uint get_size(uint cols, uint rows, uint channels, uint step) {
        return ((cols + 1) * channels + step * (rows + 1)) - 1;
    }

    VideoFilter() {
        this->RATIO = 1.0;
    }

};

#endif	/* FILTER_H */

