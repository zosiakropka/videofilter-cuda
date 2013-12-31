#include "../headers/mask.h"

void MaskFilter::process(uchar const* bytes_in, uchar* bytes_out,
        uint cols_o, uint rows_o, uint channels, uint step_in, uint step_out) {

    
    memcpy(bytes_out, bytes_in, rows_o * cols_o * channels * sizeof (uchar));

    for (uint row_i = mask.center; row_i < rows_o - mask.center; row_i++) {
        for (uint col_i = mask.center; col_i < cols_o - mask.center; col_i++) {
            size_t index = get_index(col_i, row_i, channels, step_in);
            for (uint ch = 0; ch < channels; ch++) {
                int val = 0;
                for (int x = (int) -mask.center; x <= (int) mask.center; x++) {
                    for (int y = (int) -mask.center; y <= (int) mask.center; y++) {
                        uint subindex = get_index(col_i + x, row_i + y, channels, step_in);
                        val += mask.weight(x, y) * bytes_in[subindex + ch];
                    }
                }
                bytes_out[index + ch] = (uchar) ((double) val / (double) mask.total);
            }
        }
    }
}
