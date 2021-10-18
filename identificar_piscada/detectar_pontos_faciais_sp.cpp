#include "detectar_pontos_faciais_sp.hpp"

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

void shapePradictorDetectMultiScale(dlib::shape_predictor &detector, dlib::array2d<dlib::rgb_pixel>& imagem, std::vector<cv::Rect> &rostos, std::vector<std::vector<cv::Point2f>> &pontosFaciais)
{
    std::vector<dlib::full_object_detection> pontosFaciais;
    for (unsigned long j = 0; j < rostos.size(); ++j)
    {
        //Coleta os pontos faciais de um único rosto
        dlib::full_object_detection shape = detector(imagem, rostos[j]);
        pontosFaciais.push_back(shape);
    }
}