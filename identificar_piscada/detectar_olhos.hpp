#ifndef DETECTAR_OLHOS_H
#define DETECTAR_OLHOS_H

#include "opencv2/objdetect.hpp"

cv::Ptr<cv::CascadeClassifier> iniciarDetectorOlhos();

#endif //DETECTAR_OLHOS_H