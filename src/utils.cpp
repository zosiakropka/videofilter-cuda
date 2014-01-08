#include <stdlib.h>
#include <string.h>

#include "../headers/utils.h"

Filter select_filter() {
  const char* filter = getenv("FILTER");
  if (!filter) {
    return NONE;
  } else if (!strcmp(filter, "sharpen")) {
    return SHARPEN;
  } else if (!strcmp(filter, "blur")) {
    return BLUR;
  } else if (!strcmp(filter, "resize")) {
    return RESIZE;
  } else if (!strcmp(filter, "tiltshift")) {
    return TILTSHIFT;
  }
  return NONE;
}

void parse_params(int argc, char** argv, int* threads, char** f_in, char** f_out) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " <count> <video input> <video output>" << endl;
    exit(64);
  }

  (*threads) = atoi(argv[1]);
  (*f_in) = argv[2];
  (*f_out) = argv[3];
}

void test(bool success, const char* subject, const char* file) {
  if (!success) {
    if (file) {
      cerr << "\"" << file << "\": ";
    }
    cerr << subject << " failed." << endl;
    exit(1);
  }
}

VideoProperties grab_video_properties(VideoCapture capture) {
  VideoProperties properties;
  double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
  double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  properties.frame_size.height = (int) height;
  properties.frame_size.width = (int) width;
  properties.fourcc = capture.get(CV_CAP_PROP_FOURCC);
  properties.fourcc = 0; // @TODO: change this!!!!
  properties.fps = capture.get(CV_CAP_PROP_FPS);
  properties.is_color = true; // @TODO: change this!!!!
  return properties;
}
