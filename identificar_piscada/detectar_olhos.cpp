#include "detectar_rosto.hpp"

using namespace cv;

Ptr<CascadeClassifier> initSimpleEyeDetector() {
    Ptr<CascadeClassifier> faceDetector = new CascadeClassifier;
    faceDetector->load("../../extra/haarcascade_eye.xml");

    return faceDetector;
}