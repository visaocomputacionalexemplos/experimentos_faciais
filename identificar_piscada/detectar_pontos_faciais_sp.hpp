#ifndef DETECTAR_PONTOS_FACIAIS_SP_H
#define DETECTAR_PONTOS_FACIAIS_SP_H

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "opencv2/imgproc/imgproc.hpp"

void iniciarDetectorPontosFacialSP(dlib::shape_predictor &);

void dlibHOGDetectMultiScale(
    dlib::frontal_face_detector &detector, 
    dlib::array2d<dlib::rgb_pixel>& imagem,
    std::vector<cv::Rect> &rostos);

bool shapePradictorDetectMultiScale(
        dlib::shape_predictor &detector,
        const dlib::array2d<dlib::rgb_pixel>& imagem,
        std::vector<cv::Rect> &rostos,
        std::vector<std::vector<cv::Point2f>> &pontosFaciais);

#endif //DETECTAR_PONTOS_FACIAIS_SP_H