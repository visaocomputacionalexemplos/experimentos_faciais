//https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
//Exemplo disponibilizado pela DLib: http://dlib.net/face_landmark_detection_ex.cpp.html

#include <opencv2/videoio.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

#include "detectar_pontos_faciais_sp.hpp"

using namespace std;

void coletarPontosFaciais(const dlib::array2d<dlib::rgb_pixel>& img);

// ----------------------------------------------------------------------------------------
dlib::frontal_face_detector detector;
dlib::shape_predictor shapePredictor;

int main(int argc, char **argv)
{
    // Inicia Detector faciauk
    detector = dlib::get_frontal_face_detector();

    //Carrega o modelo treinado do shape predictor
    iniciarDetectorPontosFacialSP(shapePredictor);

    //Inicia captura dos vídeos
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Video Capture Fail" << std::endl;
        return 1;
    }
    cv::Mat cam;
    cap >> cam;

    //Calcula nova dimensão da imagem para 320 pixels
    auto showSize = cv::Size(320, ((float)320 / cam.cols) * cam.rows);

    // Loop over all the images provided on the command line.
    for (;;)
    {
        //Coleta a imagem da camera
        cap >> cam;
        //Reescala a imagem para uma largura de 320 pixels
        cv::resize(cam, cam, showSize, 0, 0, cv::INTER_LINEAR_EXACT);

        //Atenção: img é apenas um invólucro pra img e não cria cópias das informações da variável "img"
        dlib::array2d<dlib::rgb_pixel> img;
        dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(cam));

        coletarPontosFaciais(img);

        cv::imshow("Origem", cam);
        cv::waitKey(5);
    }
}
    
dlib::image_window win, win_faces;
std::vector<dlib::rectangle> rostosDetectados;

void coletarPontosFaciais(const dlib::array2d<dlib::rgb_pixel>& imagemOriginal)
{
    //Detecta os pontos faciais e retorna a lista de rostos detectados
    rostosDetectados = detector(imagemOriginal);
    //cout << "Numero de rostos detectados: " << dets.size() << endl;

    //Percorre os rostos detectados e coloca os pontos faciais
    std::vector<dlib::full_object_detection> pontosFaciais;
    for (unsigned long j = 0; j < rostosDetectados.size(); ++j)
    {
        //Coleta os pontos faciais de um único rosto
        dlib::full_object_detection shape = shapePredictor(imagemOriginal, rostosDetectados[j]);
        pontosFaciais.push_back(shape);
    }

    // Exibe a imagem com o contorno dos pontos.
    win.clear_overlay();
    win.set_image(imagemOriginal);
    win.add_overlay(render_face_detections(pontosFaciais));

    // Recorta o rosto da imagem original
    dlib::array<dlib::array2d<dlib::rgb_pixel>> rostosRecortados;
    extract_image_chips(imagemOriginal, get_face_chip_details(pontosFaciais), rostosRecortados);
    win_faces.set_image(tile_images(rostosRecortados));
}