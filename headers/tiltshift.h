/* 
 * File:   tiltshift.h
 * Author: Zosia Sobocinska
 *
 * Created on December 26, 2013, 6:29 PM
 */

#ifndef TILTSHIFT_FILTER_H
#define	TILTSHIFT_FILTER_H

#include "filter.h"
#include "blur.h"
#include <math.h>

#ifndef GPU
#endif

class TiltshiftFilter : public VideoFilter {
private:
  BlurFilter blur_filter;

public:

  void process(uchar const* bytes_in, uchar* bytes_out,
          uint cols, uint rows, uint channels, uint step_in, uint step_out);

  TiltshiftFilter() : VideoFilter() {
    strcpy(name, "tiltshift");
  }

  void alloc_buffers();
  void free_buffers();

};

#endif	/* TILTSHIFT_FILTER_H */

