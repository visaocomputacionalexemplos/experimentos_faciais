#include "piscadas_facemark_lbf.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> initFacemarkAAM() {
    Ptr<Facemark> facemark = FacemarkAAM::create();
    //facemark->loadModel("../../extra/lbfmodel.yaml");

    return facemark;
}