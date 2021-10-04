#include "piscadas_facemark_lbf.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> initFacemarkLBF() {
    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel("../../extra/lbfmodel.yaml");

    return facemark;
}