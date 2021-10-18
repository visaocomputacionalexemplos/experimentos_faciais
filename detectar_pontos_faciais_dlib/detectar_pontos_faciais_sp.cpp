#include "detectar_pontos_faciais_sp.hpp"


void iniciarDetectorPontosFacialSP(dlib::shape_predictor &sp)
{
    dlib::deserialize("../../extra/shape_predictor_68_face_landmarks.dat") >> sp;
}