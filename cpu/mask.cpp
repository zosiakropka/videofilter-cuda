#include "../headers/mask.h"

void MaskFilter::alloc_buffers() {
  VideoFilter::alloc_buffers();
}

void MaskFilter::free_buffers() {
  VideoFilter::free_buffers();
}

void MaskFilter::process(uchar const* bytes_in, uchar* bytes_out,
    uint cols, uint rows, uint channels, uint step_in, uint step_out) {

  for (uint row = 0; row < rows; row++) {
    for (uint col = 0; col < cols; col++) {
      uint index = get_index(col, row, channels, step_in);
      for (uint ch = 0; ch < channels; ch++) {
        int val = 0;
        for (int x = (int) -mask.center; x <= (int) mask.center; x++) {
          for (int y = (int) -mask.center; y <= (int) mask.center; y++) {
            uint subindex = get_index(col + x, row + y, channels, step_in);
            val += mask.weight(x, y) * bytes_in[subindex + ch];
          }
        }
        bytes_out[index + ch] = (uchar) ((double) val / (double) mask.total);
      }
    }
  }
}
