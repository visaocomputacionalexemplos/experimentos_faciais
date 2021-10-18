#include "detectar_pontos_faciais_sp.hpp"
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <iostream>

void iniciarDetectorPontosFacialSP(dlib::shape_predictor &sp)
{
    dlib::deserialize("../../extra/shape_predictor_68_face_landmarks.dat") >> sp;
}

void dlibHOGDetectMultiScale(dlib::frontal_face_detector &detector, dlib::array2d<dlib::rgb_pixel>& imagem, std::vector<cv::Rect> &rostos)
{
    for (dlib::rectangle &rect : detector(imagem))
    {
        rostos.push_back(cv::Rect(rect.left(), rect.top(), rect.width(), rect.height()));
    }
}

bool shapePradictorDetectMultiScale(dlib::shape_predictor &detector, const dlib::array2d<dlib::rgb_pixel>& imagem, std::vector<cv::Rect> &rostos, std::vector<std::vector<cv::Point2f>> &pontosFaciais)
{
    for (unsigned long j = 0; j < rostos.size(); ++j)
    {
        //Convert de opencv rect pra dlib rect
        dlib::rectangle rectangleDlib(
            (long)rostos[j].tl().x, (long)rostos[j].tl().y, 
            (long)rostos[j].br().x - 1, (long)rostos[j].br().y - 1);
        //Coleta os pontos faciais de um único rosto
        dlib::full_object_detection shape = detector(imagem, rectangleDlib);

        //Converte para cv::Point2f
        std::vector<cv::Point2f> pontosOpenCV;

        if (shape.num_parts()) {
            for (int k=0; k<shape.num_parts(); k++) {
                pontosOpenCV.push_back(
                    cv::Point(shape.part(k).x(), shape.part(k).y())
                );
            }
            //pontosFaciais.push_back(shape);
            pontosFaciais.push_back(pontosOpenCV);
        }
    }
    return pontosFaciais.size();
}