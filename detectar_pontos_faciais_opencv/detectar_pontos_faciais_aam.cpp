#include <opencv2/opencv.hpp>

#include "detectar_pontos_faciais_lbf.hpp"
#include "utils.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> iniciarDetectorPontosFacialAAM()
{
    Ptr<FacemarkAAM> facemark = FacemarkAAM::create();
    facemark->loadModel("../../extra/aam_model.yaml");
    return facemark;
}

bool facemarkAAMFit(FacemarkAAM *ammFacemark, Ptr<CascadeClassifier> eyeDetector, Mat imagemOriginal, std::vector<Rect> rostosDetectados, std::vector<std::vector<Point2f>> &pontosFaciais)
{
    std::vector<FacemarkAAM::Config> conf;
    std::vector<Rect> faces_fit;
    float scale;
    Point2f T;
    Mat R;
    FacemarkAAM::Data data;
    ammFacemark->getData(&data);
    std::vector<Point2f> s0 = data.s0;
    FacemarkAAM::Params params;
    params.scales.clear();
    params.scales.push_back(2);
    params.scales.push_back(4);

    for (unsigned long j = 0; j < rostosDetectados.size(); j++)
    {
        if (getInitialFitting(imagemOriginal, rostosDetectados[j], s0, eyeDetector, R, T, scale))
        {
            conf.push_back(FacemarkAAM::Config(R, T, scale, (int)params.scales.size() - 1));
            //conf.push_back(FacemarkAAM::Config(R,T,scale));
            faces_fit.push_back(rostosDetectados[j]);
        }
    }

    if (conf.size() > 0)
    {
        ammFacemark->fitConfig(imagemOriginal, faces_fit, pontosFaciais, conf);
        return true;
    }
    return false;
}