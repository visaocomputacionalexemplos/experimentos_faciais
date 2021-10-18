#include "detectar_pontos_faciais_lbf.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> iniciarDetectorPontosFacialLBF() {
    Ptr<Facemark> facemarkOpenCV = FacemarkLBF::create();
    facemarkOpenCV->loadModel("../../extra/lbfmodel.yaml");

    return facemarkOpenCV;
}