#include "detectar_rosto.hpp"

using namespace cv;

Ptr<CascadeClassifier> initSimpleFaceDetector() {
    Ptr<CascadeClassifier> faceDetector = new CascadeClassifier;
    faceDetector->load("../../extra/haarcascade_frontalface_alt2.xml");

    return faceDetector;
}