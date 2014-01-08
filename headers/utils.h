/* 
 * File:   utils.h
 * Author: Zosia Sobocinska
 *
 * Created on December 26, 2013, 12:12 PM
 */

#ifndef UTILS_H
#define	UTILS_H

#include <iostream>
#include <cstdlib>

using std::cerr;
using std::endl;

#include <opencv2/opencv.hpp>
using cv::Mat;
using cv::VideoCapture;
using cv::VideoWriter;
using cv::Size;

enum Filter {
  NONE,
  SHARPEN,
  BLUR,
  RESIZE,
  TILTSHIFT,
};

const uint FILTERS_COUNT = 4;

Filter select_filter();

void parse_params(int argc, char** argv, int* threads, char** f_in, char** f_out);

void test(bool success, const char* subject = "unknown", const char* file = NULL);

struct VideoProperties {
  int fourcc;
  double fps;
  Size frame_size;
  bool is_color;
};

VideoProperties grab_video_properties(VideoCapture capture);

#endif	/* UTILS_H */

