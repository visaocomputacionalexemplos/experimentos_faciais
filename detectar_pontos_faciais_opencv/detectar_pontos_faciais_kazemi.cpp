#include "detectar_pontos_faciais_lbf.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> iniciarDetectorPontosFacialKazemi() {
    Ptr<Facemark> facemark = FacemarkKazemi::create();
    facemark->loadModel("../../extra/kazemi_model.dat");

    return facemark;
}