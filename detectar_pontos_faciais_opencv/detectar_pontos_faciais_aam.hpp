#ifndef DETECTAR_PONTOS_FACIAIS_AAM_H
#define DETECTAR_PONTOS_FACIAIS_AAM_H

#include "opencv2/face.hpp"

cv::Ptr<cv::face::Facemark> iniciarDetectorPontosFacialAAM();


bool facemarkAAMFit(cv::face::FacemarkAAM* facemark, cv::Ptr<cv::CascadeClassifier> eyeDetector, cv::Mat imagemOriginal, std::vector<cv::Rect> rostosDetectados, std::vector<std::vector<cv::Point2f>>& pontosFaciais);

#endif //DETECTAR_PONTOS_FACIAIS_AAM_H