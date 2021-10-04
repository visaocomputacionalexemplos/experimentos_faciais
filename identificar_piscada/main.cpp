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

#include "utils.hpp"
#include "detectar_rosto.hpp"
#include "piscadas_facemark_lbf.hpp"

using namespace cv;
using namespace std;
using namespace ml;
using namespace cv::face;

Ptr<Facemark> facemark;
Ptr<CascadeClassifier> faceDetector;

void processar(Mat img);

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

const auto liminarOlhoFechado = 0.20;
void processar(Mat imagemOriginal)
{
    olhoDimencoes olhoEsquerdoDimencoes, olhoDireitoDimencoes;
    vector<Rect> rostosDetectados;
    vector<vector<Point2f>> pontosFaciais;
    Mat imagemComPontosFaciais;
    bool olhoEsquerdoAberto = false, olhoDireitoAberto = false;
    int piscadas = 0;

    //Chama funcao que detecta os rostos
    faceDetector->detectMultiScale(imagemOriginal, rostosDetectados);

    if (rostosDetectados.size() != 0)
    {
        imagemComPontosFaciais = imagemOriginal.clone();

        for (auto &&rostoDetec : rostosDetectados)
        {
            demarcarRostoDetectado(imagemOriginal, rostoDetec);
        }

        //Detecta os pontos faciais
        if (facemark->fit(imagemOriginal, rostosDetectados, pontosFaciais))
        {
            demarcarPontosFaciais(imagemComPontosFaciais, rostosDetectados, pontosFaciais);
            tracejarRegiaoInteresse(imagemComPontosFaciais, pontosFaciais);
            demarcarContornoOlhos(imagemOriginal, pontosFaciais);

            for (unsigned long i = 0; i < rostosDetectados.size(); i++)
            {
                //Coleta as dimensões(largura, altura e proporção) dos olhos
                std::tie(olhoEsquerdoDimencoes, olhoDireitoDimencoes) = coletarDimensoesOlhos(pontosFaciais[i]);
                escreverDimensoesOlhos(imagemOriginal, rostosDetectados[i], olhoEsquerdoDimencoes, olhoDireitoDimencoes);

                //Olho direito esta aberto?
                if (olhoEsquerdoDimencoes.proporcao > liminarOlhoFechado) {
                    //Registra que olho esta aberto
                    olhoEsquerdoAberto = true;
                } else  {
                    //Verifica se último registro é de olho aberto
                    if (olhoEsquerdoAberto) {
                        //Se for contabiliza piscada
                        piscadas++;
                    }
                    olhoEsquerdoAberto = false;
                }

                //Olho direito esta aberto?
                if (olhoDireitoDimencoes.proporcao > liminarOlhoFechado) {
                    //Registra que olho esta aberto
                    olhoDireitoAberto = true;
                } else  {
                    //Verifica se último registro é de olho aberto
                    if (olhoDireitoAberto) {
                        //Se for contabiliza piscada
                        piscadas++;
                    }
                    olhoDireitoAberto = false;
                }
            }
        }

        imshow("Pontos faciais", imagemComPontosFaciais);
        waitKey(5);
    }
    else
    {
        cout << "Nenhum rosto detectado." << endl;
    }
}