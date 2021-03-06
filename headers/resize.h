/* 
 * File:   resize.h
 * Author: Zosia Sobocinska
 *
 * Created on December 30, 2013, 10:28 PM
 */

#ifndef RESIZE_FILTER_H
#define	RESIZE_FILTER_H

#include "filter.h"

#ifndef GPU
#endif

class ResizeFilter : public VideoFilter {
private:
  static const int NEIGHBOURHOOD = 2;
  static const double A = 1.0;

  //  __device__ static double get_weight(double distance_x, double distance_y);
  //  __device__ static double get_kernel(double distance);

  double get_distance(int in, int out);

public:

  ResizeFilter() : VideoFilter() {
    strcpy(name, "resize");
  }
  void process(uchar const* bytes_in, uchar* bytes_out,
          uint cols_in, uint rows_in, uint channels, uint step_in, uint step_out);
};

#endif	/* RESIZE_FILTER_H */

