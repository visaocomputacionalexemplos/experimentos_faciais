//http://visaocomputacional.com.br/deteccao-de-pontos-faciais-facemark-com-opencv-e-dlib/
//https://github.com/visaocomputacionalexemplos/experimentos_faciais

//FacemarkKazemi https://www.csc.kth.se/~vahidk/face_ert.html
//FacemarkAAM: https://ibug.doc.ic.ac.uk/media/uploads/documents/tzimiro_pantic_iccv2013.pdf
//FacemarkLBF: http://www.jiansun.org/papers/CVPR14_FaceAlignment.pdf

#include <opencv2/opencv.hpp>
#include <iostream>

#include "utils.hpp"
#include "detectar_rosto.hpp"
#include "detectar_olhos.hpp"
#include "detectar_pontos_faciais_lbf.hpp"
#include "detectar_pontos_faciais_aam.hpp"
#include "detectar_pontos_faciais_kazemi.hpp"

//Detectores
cv::Ptr<cv::CascadeClassifier> faceDetector;
cv::Ptr<cv::CascadeClassifier> eyeDetector;
cv::Ptr<cv::face::Facemark> facemark;

bool fitEmTonsDeCinza;
bool requerDetecaoDosOlhos;
std::string tipo;

void coletarPontosFaciais(cv::Mat img);

int main()
{
    std::cout << "Informe o tipo de detector por pontos faciais que você deseja testar:" << std::endl
              << " 1: LBF" << std::endl
              << " 2: AAM" << std::endl
              << " 3: Kazemi" << std::endl;

    std::cin >> tipo;

    //Inicia marcados de pontos faciais
    if (tipo.compare("1") == 0)
    {
        //LBF
        fitEmTonsDeCinza = true;
        requerDetecaoDosOlhos = false;
        facemark = iniciarDetectorPontosFacialLBF();
    }
    else if (tipo.compare("2") == 0)
    {
        //AAM
        fitEmTonsDeCinza = false;
        requerDetecaoDosOlhos = true;
        facemark = iniciarDetectorPontosFacialAAM();
    }
    else if (tipo.compare("3") == 0)
    {
        //Kazemi
        fitEmTonsDeCinza = false;
        requerDetecaoDosOlhos = false;
        facemark = iniciarDetectorPontosFacialKazemi();
    }
    else
    {
        std::cout << "Nenhum tipo informado." << std::endl;
    }

    //Inicia detector facial e de olhos por Haarcascade
    faceDetector = iniciarDetectorFacial();
    if (requerDetecaoDosOlhos)
        eyeDetector = iniciarDetectorOlhos();

    //Inicia captura dos vídeos
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Video Capture Fail" << std::endl;
        return 1;
    }
    cv::Mat img;
    cap >> img;

    //Calcula nova dimensão da imagem para 480 pixels
    auto showSize = cv::Size(480, ((float)480 / img.cols) * img.rows);

    for (;;)
    {
        //Coleta a imagem da camera
        cap >> img;

        //Reescala a imagem para uma largura de 320 pixels
        cv::resize(img, img, showSize, 0, 0, cv::INTER_LINEAR_EXACT);

        coletarPontosFaciais(img);
        cv::imshow("Origem", img);

        cv::waitKey(5);
    }
}

cv::Mat imagemOriginalCinza;
cv::Mat imagemComPontosFaciais;
std::vector<cv::Rect> rostosDetectados;
std::vector<std::vector<cv::Point2f>> pontosFaciais;
bool pontosDetectados = false;

void coletarPontosFaciais(cv::Mat imagemOriginal)
{
    rostosDetectados.clear();
    pontosFaciais.clear();

    {
        //Converte em tons de cinza e equaliza a imagem
        //Detecção por haarcascade funcionam bem com imagens equalizadas
        cvtColor(imagemOriginal, imagemOriginalCinza, cv::COLOR_BGR2GRAY);
        equalizeHist(imagemOriginalCinza, imagemOriginalCinza);

        //Detecta os rostos na imagem
        faceDetector->detectMultiScale(imagemOriginal, rostosDetectados);
    }

    if (rostosDetectados.size() != 0)
    {
        imagemComPontosFaciais = imagemOriginal.clone();

        //Demarca rosto na imagem original
        for (auto &&rostoDetec : rostosDetectados)
        {
            demarcarRostoDetectado(imagemOriginal, rostoDetec);
        }

        if (requerDetecaoDosOlhos)
        {
            //Detecta os pontos faciais com código personalizado para o algorítmo AAM
            pontosDetectados = facemarkAAMFit(static_cast<cv::face::FacemarkAAM *>(facemark.get()), eyeDetector,
                           fitEmTonsDeCinza
                               ? imagemOriginalCinza
                               : imagemOriginal,
                           rostosDetectados,
                           pontosFaciais);
        }
        else
        {
            //Detecta os pontos faciais
            pontosDetectados = facemark->fit(fitEmTonsDeCinza
                                                 ? imagemOriginalCinza
                                                 : imagemOriginal,
                                             rostosDetectados, pontosFaciais);
        }

        if (pontosDetectados)
        {
            demarcarPontosFaciais(imagemComPontosFaciais, rostosDetectados, pontosFaciais);
            imshow("Pontos faciais", imagemComPontosFaciais);
        }
    }
}