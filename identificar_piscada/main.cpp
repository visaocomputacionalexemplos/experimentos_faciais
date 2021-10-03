//FacemarkKazemi https://www.csc.kth.se/~vahidk/face_ert.html
//FacemarkAAM: https://ibug.doc.ic.ac.uk/media/uploads/documents/tzimiro_pantic_iccv2013.pdf
//FacemarkLBF: http://www.jiansun.org/papers/CVPR14_FaceAlignment.pdf

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>

#include "piscadas_facemark_lbf.hpp"
#include "detectar_rosto.hpp"

using namespace cv;
using namespace std;
using namespace ml;
using namespace cv::face;

Ptr<Facemark> facemark;
Ptr<CascadeClassifier> faceDetector;

void processar(Mat img);
void demarcarRostoDetectado(Mat img, const Rect &regiao);
void demarcarPontosFaciais(Mat img, const vector<Rect> &rostosDetectados, const vector<vector<Point2f>> &pontosFaciais);
void demarcarContornoOlhos(Mat img, const vector<vector<Point2f>> &pontosFaciais);

int main()
{
    //Inicia detector de face por Haarcascade
    faceDetector = initSimpleFaceDetector();

    //Inicia marcados de pontos faciais
    facemark = initFacemarkLBF();

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "Video Capture Fail" << endl;
        return 1;
    }

    Mat img;
    cap >> img;

    //Calcula a escala e coleta a quantidade de linhas para uma imagem de 640 cols
    auto showSize = Size(640, (640 / img.cols) * img.rows);

    for (;;)
    {
        //Coleta a imagem da camera
        cap >> img;

        //Escala a imagem, converte para tons de cinza
        resize(img, img, showSize, 0, 0, INTER_LINEAR_EXACT);
        cvtColor(img, img, COLOR_BGR2GRAY);

        processar(img);
        imshow("Origem", img);

        waitKey(5);
    }
}

void processar(Mat imagemOriginal)
{
    vector<Rect> rostosDetectados;
    vector<vector<Point2f>> pontosFaciais;
    Mat rostoComPontosFaciais;

    //Chama funcao que detecta os rostos
    faceDetector->detectMultiScale(imagemOriginal, rostosDetectados);

    if (rostosDetectados.size() != 0)
    {
        rostoComPontosFaciais = imagemOriginal.clone();

        for (auto &&rostoDetec : rostosDetectados)
        {
            demarcarRostoDetectado(imagemOriginal, rostoDetec);
        }

        //Detecta os pontos faciais
        if (facemark->fit(imagemOriginal, rostosDetectados, pontosFaciais))
        {
            demarcarPontosFaciais(rostoComPontosFaciais, rostosDetectados, pontosFaciais);
            demarcarContornoOlhos(imagemOriginal, pontosFaciais);
        }

        imshow("Pontos faciais", rostoComPontosFaciais);
        waitKey(5);
    }
    else
    {
        cout << "Nenhum rosto detectado." << endl;
    }
}

void demarcarRostoDetectado(Mat img, const Rect &regiao)
{
    cv::rectangle(img, regiao, Scalar(200, 0, 0));
}

void demarcarPontosFaciais(Mat img, const vector<Rect> &rostosDetectados, const vector<vector<Point2f>> &pontosFaciais)
{
    for (unsigned long i = 0; i < rostosDetectados.size(); i++)
    {
        for (auto &&ponto : pontosFaciais[i])
        {
            cv::circle(img, ponto, 2, cv::Scalar(0, 0, 0), FILLED);
        }
    }
}

void demarcarContornoOlhos(Mat img, const vector<vector<Point2f>> &pontosFaciais)
{
    //Pontos olhos esquerdo 36 39 41
    //Pontos olhos direito  42 45 47
    for (auto &&pontosDeUmaFace : pontosFaciais)
    {
        //Contorna olho esquerdo
        for (int x = 36; x < 41; x++)
        {
            cv::line(img, pontosDeUmaFace[x], pontosDeUmaFace[x+1], cv::Scalar(200, 0, 0));
        }
        cv::line(img, pontosDeUmaFace[41], pontosDeUmaFace[36], cv::Scalar(200, 0, 0));

        //Contorna olho direito
        for (int x = 42; x < 47; x++)
        {
            cv::line(img, pontosDeUmaFace[x], pontosDeUmaFace[x+1], cv::Scalar(200, 0, 0));
        }
        cv::line(img, pontosDeUmaFace[42], pontosDeUmaFace[47], cv::Scalar(200, 0, 0));
    }
}
