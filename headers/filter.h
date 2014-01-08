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

#include "gpu.h"

typedef unsigned char uchar;

class VideoFilter {
protected:

  char name[50];

  uchar **buffers;
  uint buffers_count;
  double RATIO;

  uint threads;

  int cols_in, rows_in;
  int step_in, step_out;
  uint channels;

  inline int cols_out() {
    #define frame_cols_out(ratio, cols_in) (int) (ratio * (double) cols_in)
    return frame_cols_out(RATIO, cols_in);
  }

  inline int rows_out() {
    #define frame_rows_out(ratio, rows_in) (int) (ratio * (double) rows_in)
    return frame_rows_out(RATIO, rows_in);
  }

  inline uint size_in() {
    #define frame_size_in(rows_in, step_in) (rows_in * step_in)
    return frame_size_in(rows_in, step_in);
  };

  inline uint size_out() {
    #define frame_size_out(rows_in, step_out, ratio) (frame_rows_out(ratio, rows_in) * step_out)
    return frame_size_out(rows_in, step_out, RATIO);
  };

public:

  inline char* get_name() {
    return name;
  }

  virtual void alloc_buffers();
  virtual void free_buffers();

  void filter(VideoCapture v_in, VideoWriter v_out);

  inline void set_ratio(double ratio) {
    this->RATIO = ratio;
  }

  inline void set_threads(uint threads) {
    this->threads = threads;
  }

  virtual void process(uchar const* bytes_in, uchar* bytes_out,
          uint cols, uint rows, uint channels, uint step_in, uint step_out) {
  };

  static inline uint get_index(int col, int row, uint channels, uint step) {
    #define array_index(col, row, channels, step) ((col) * (channels) + (step) * (row))
    col = (col < 0) ? 0 : col;
    row = (row < 0) ? 0 : row;
    return col * channels + step * row;
  }

  VideoFilter() {
    this->RATIO = 1.0;
    buffers = NULL;
    buffers_count = 0;
  }

};

#endif	/* FILTER_H */

