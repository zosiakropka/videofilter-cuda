#include "../headers/tiltshift.h"

void TiltshiftFilter::process(uchar const* bytes_in, uchar* bytes_out,
        uint cols, uint rows, uint channels, uint step_in, uint step_out) {

    blur_filter.process(bytes_in, bytes_out, cols, rows, channels, step_in, step_out);

    for (uint row_i = 0; row_i < rows; row_i++) {

        double opacity = 1.0 - fabs(1.0 - (double) row_i / (double) (rows / 2));
        //opacity = (1.0 - opacity * opacity);
        opacity = (opacity > 1.0) ? 1.0 : opacity;

        double opacity_alt = 1.0 - opacity;

        for (uint col_i = 0; col_i < cols; col_i++) {
            size_t index = get_index(col_i, row_i, channels, step_in);
            for (uint ch = 0; ch < channels; ch++) {
                uchar val = (uchar) (opacity * (double) bytes_in[index + ch] +
                        opacity_alt * (double) bytes_out[index + ch]);
                val = (val < 256) ? val : 255;

                bytes_out[index + ch] = val;
            }
        }

    }
}
