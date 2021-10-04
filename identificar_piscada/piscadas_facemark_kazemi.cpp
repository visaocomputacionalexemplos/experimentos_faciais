#include "piscadas_facemark_lbf.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> initFacemarkKazemi() {
    Ptr<Facemark> facemark = FacemarkKazemi::create();
    facemark->loadModel("../../extra/kazemi_model.dat");

    return facemark;
}