#include <opencv2/videoio.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>

#include "utils.hpp"
#include "detectar_rosto.hpp"
#include "detectar_pontos_faciais_lbf.hpp"
#include "detectar_pontos_faciais_sp.hpp"

dlib::frontal_face_detector faceDetectorDlib;
dlib::shape_predictor facemarkDlib;
cv::Ptr<cv::CascadeClassifier> faceDetectorOpenCV;
cv::Ptr<cv::face::Facemark> facemarkOpenCV;

bool detectarRostosComDlib;
bool detectarPontosComDlib;

std::string tipo;

void coletarPontosFaciais(cv::Mat img);

int main()
{
    std::cout << "Detector Facial: Informe o tipo de detector de rostos que você deseja testar:" << std::endl
              << " 1: OpenCV - Haarscascade" << std::endl
              << " 2: Dlib - HoG Face Detector" << std::endl;
    std::cin >> tipo;
    //Inicia marcados de pontos faciais
    if (tipo.compare("1") == 0)
        detectarRostosComDlib = false; //Opencv LBF
    else if (tipo.compare("2") == 0)
        detectarRostosComDlib = true; //DLIB - Shape Predict
    else
    {
        std::cout << "Nenhum tipo informado." << std::endl;
        return 1;
    }

    std::cout << "Detector Pontos Faciais: Informe o tipo de detector por pontos faciais que você deseja testar:" << std::endl
              << " 1: OpenCV - LBF" << std::endl
              << " 2: Dlib - Shape Predict" << std::endl;
    std::cin >> tipo;
    //Inicia marcados de pontos faciais
    if (tipo.compare("1") == 0)
        detectarPontosComDlib = false; //Opencv LBF
    else if (tipo.compare("2") == 0)
        detectarPontosComDlib = true; //DLIB - Shape Predict
    else
    {
        std::cout << "Nenhum tipo informado." << std::endl;
        return 1;
    }

    //Inicia detector facial e de olhos por Haarcascade
    if (detectarRostosComDlib)
        faceDetectorDlib = dlib::get_frontal_face_detector();
    else
        faceDetectorOpenCV = iniciarDetectorFacial();

    if (detectarPontosComDlib)
        //Carrega o modelo treinado do dlib shape predictor
        iniciarDetectorPontosFacialSP(facemarkDlib);
    else
        //Carrega o modelo treinado para o opencv lbf
        facemarkOpenCV = iniciarDetectorPontosFacialLBF();

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

dlib::array2d<dlib::rgb_pixel> imagemOriginalDlib;
cv::Mat imagemOriginalCinza;
const auto liminarOlhoFechado = 0.24;
const auto liminarOlhoAberto = 0.26;
bool olhoEsquerdoAberto = false, olhoDireitoAberto = false, pontosDetectados = false;
int piscadas = 0;

void coletarPontosFaciais(cv::Mat imagemOriginal)
{
    std::vector<cv::Rect> rostosDetectados;
    olhoDimencoes olhoEsquerdoDimencoes, olhoDireitoDimencoes;
    std::vector<std::vector<cv::Point2f>> pontosFaciais;
    cv::Mat imagemComPontosFaciais;

    if (detectarRostosComDlib || detectarPontosComDlib) {
        //Converte a imagem do opencv na dlib
        dlib::assign_image(imagemOriginalDlib, dlib::cv_image<dlib::bgr_pixel>(imagemOriginal));
    }

    {
        //Converte em tons de cinza e equaliza a imagem
        //Detecção por haarcascade funcionam bem com imagens equalizadas
        cvtColor(imagemOriginal, imagemOriginalCinza, cv::COLOR_BGR2GRAY);
        equalizeHist(imagemOriginalCinza, imagemOriginalCinza);

        //Detecta os rostos na imagem
        if (detectarRostosComDlib)
        {
            dlibHOGDetectMultiScale(faceDetectorDlib, imagemOriginalDlib, rostosDetectados);
        }
        else
        {
            faceDetectorOpenCV->detectMultiScale(imagemOriginal, rostosDetectados);
        }
    }

    if (rostosDetectados.size() != 0)
    {
        imagemComPontosFaciais = imagemOriginal.clone();

        //Demarca rosto na imagem original
        for (auto &&rostoDetec : rostosDetectados)
        {
            demarcarRostoDetectado(imagemOriginal, rostoDetec);
        }

        //Detecta os pontos faciais
        if (detectarPontosComDlib)
        {
            pontosDetectados = shapePradictorDetectMultiScale(facemarkDlib, imagemOriginalDlib, rostosDetectados, pontosFaciais);
        }
        else
        {
            pontosDetectados = facemarkOpenCV->fit(imagemOriginal, rostosDetectados, pontosFaciais);
        }

        if (pontosDetectados)
        {
            demarcarPontosFaciais(imagemComPontosFaciais, rostosDetectados, pontosFaciais);

            for (unsigned long i = 0; i < rostosDetectados.size(); i++)
            {
                //Coleta as dimensões(largura, altura e proporção) dos olhos
                std::tie(olhoEsquerdoDimencoes, olhoDireitoDimencoes) = coletarDimensoesOlhos(pontosFaciais[i]);
                escreverDimensoesOlhos(imagemOriginal, rostosDetectados[i], olhoEsquerdoDimencoes, olhoDireitoDimencoes);

                //Olho direito esta aberto?
                if (olhoEsquerdoDimencoes.proporcao > liminarOlhoAberto)
                {
                    //Registra que olho esta aberto
                    olhoEsquerdoAberto = true;
                }
                else if (olhoEsquerdoDimencoes.proporcao < liminarOlhoFechado)
                {
                    //Verifica se último registro é de olho aberto
                    if (olhoEsquerdoAberto)
                    {
                        //Se for contabiliza piscada
                        piscadas++;
                    }
                    olhoEsquerdoAberto = false;
                }

                //Olho direito esta aberto?
                if (olhoDireitoDimencoes.proporcao > liminarOlhoFechado)
                {
                    //Registra que olho esta aberto
                    olhoDireitoAberto = true;
                }
                else
                {
                    //Verifica se último registro é de olho aberto
                    if (olhoDireitoAberto)
                    {
                        //Se for contabiliza piscada
                        piscadas++;
                    }
                    olhoDireitoAberto = false;
                }
            }

            imshow("Pontos faciais", imagemComPontosFaciais);
        }
    }

    escreverQtdPiscadas(imagemOriginal, piscadas);
}