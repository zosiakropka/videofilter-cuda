#include "../headers/filter.h"
#include "../headers/blur.h"

void VideoFilter::filter(VideoCapture v_in, VideoWriter v_out) {

  Mat frame_in, frame_out;

  cols_in = v_in.get(CV_CAP_PROP_FRAME_WIDTH);
  rows_in = v_in.get(CV_CAP_PROP_FRAME_HEIGHT);

  Mat testframe_in, testframe_out;
  v_in >> testframe_in;
  testframe_out.create(rows_out(), cols_out(), testframe_in.type());

  channels = testframe_in.channels();
  step_in = testframe_in.step;
  step_out = testframe_out.step;

  v_in.set(CV_CAP_PROP_POS_FRAMES, 0);

  alloc_buffers();

  while (true) {

    v_in >> frame_in;

    if (frame_in.empty()) {
      break;
    }

    frame_out.create(rows_out(), cols_out(), frame_in.type());

    uchar* bytes_in = (uchar*) (frame_in.data);
    uchar* bytes_out = (uchar*) (frame_out.data);

    process(bytes_in, bytes_out, cols_in, rows_in, channels, step_in, step_out);

    v_out << frame_out;


  }

  free_buffers();
}

void VideoFilter::alloc_buffers() {
  #if CUDA
  buffers_count = 2;
  buffers = new uchar* [buffers_count];
  dev::alloc_mem_uchar(&buffers[0], size_in());
  dev::alloc_mem_uchar(&buffers[1], size_out());
  #endif
}

void VideoFilter::free_buffers() {
  #if CUDA
  for (uint i = 0; i < buffers_count; i++) {
    dev::free_mem_uchar(&(buffers[i]));
  }
  #endif
}
