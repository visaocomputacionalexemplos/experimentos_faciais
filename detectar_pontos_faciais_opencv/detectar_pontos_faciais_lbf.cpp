#include "detectar_pontos_faciais_lbf.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> iniciarDetectorPontosFacialLBF() {
    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel("../../extra/lbfmodel.yaml");

    return facemark;
}