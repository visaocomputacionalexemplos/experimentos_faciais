#include "detectar_rosto.hpp"

using namespace cv;

Ptr<CascadeClassifier> iniciarDetectorFacial() {
    Ptr<CascadeClassifier> faceDetectorOpenCV = new CascadeClassifier;
    faceDetectorOpenCV->load("../../extra/haarcascade_frontalface_alt2.xml");

    return faceDetectorOpenCV;
}