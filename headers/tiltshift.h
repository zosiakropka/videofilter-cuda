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

class TiltshiftFilter : public VideoFilter {
private:
    BlurFilter blur_filter;

public:

    void process(uchar const* bytes_in, uchar* bytes_out,
            uint cols, uint rows, uint channels, uint step_in, uint step_out);
};

#endif	/* TILTSHIFT_FILTER_H */

