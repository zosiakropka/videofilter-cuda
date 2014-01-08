/* 
 * File:   main.cpp
 * Author: Zosia Sobocinska
 *
 * Created on December 25, 2013, 11:49 PM
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "../headers/utils.h"
#include "../headers/filter.h"
#include "../headers/tiltshift.h"
#include "../headers/blur.h"
#include "../headers/sharpen.h"
#include "../headers/resize.h"

using cv::VideoCapture;
using cv::VideoWriter;

/*
 * 
 */
int main(int argc, char** argv) {


  char* f_in;
  char* f_out;
  int threads;

  parse_params(argc, argv, &threads, &f_in, &f_out);

  Filter filter_id = select_filter();

  double ratio = 2;

  VideoCapture v_in(f_in);
  test(v_in.isOpened(), "open", f_in);
  VideoProperties p = grab_video_properties(v_in);
  VideoWriter v_out;

  if (filter_id == RESIZE) {
    p.frame_size.width = (int) (ratio * (double) (p.frame_size.width));
    p.frame_size.height = (int) (ratio * (double) (p.frame_size.height));
  }

  v_out.open(f_out, p.fourcc, p.fps, p.frame_size, p.is_color);

  VideoFilter* filter;
  test(v_out.isOpened(), "open", f_out);
  switch (filter_id) {
    case NONE:
      break;
    case BLUR: filter = new BlurFilter;
      break;
    case SHARPEN: filter = new SharpenFilter;
      break;
    case RESIZE: filter = new ResizeFilter;
      filter->set_ratio(ratio);
      break;
    case TILTSHIFT: filter = new TiltshiftFilter;
      break;
  }

  filter->set_threads(threads);

  filter->filter(v_in, v_out);

  return 0;
}

