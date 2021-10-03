#ifndef DETECTAR_ROSTO_H
#define DETECTAR_ROSTO_H

#include "opencv2/objdetect.hpp"

cv::Ptr<cv::CascadeClassifier> initSimpleFaceDetector();

#endif //DETECTAR_ROSTO_H