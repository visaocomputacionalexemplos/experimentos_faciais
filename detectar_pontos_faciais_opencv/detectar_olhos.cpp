#include "detectar_rosto.hpp"

using namespace cv;

Ptr<CascadeClassifier> iniciarDetectorOlhos() {
    Ptr<CascadeClassifier> faceDetector = new CascadeClassifier;
    faceDetector->load("../../extra/haarcascade_eye.xml");

    return faceDetector;
}