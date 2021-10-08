/*#include <stdio.h>
#include <fstream>
#include <sstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <ctime>

using namespace std;
using namespace cv;
using namespace cv::face;

bool myDetector( InputArray image, OutputArray ROIs, CascadeClassifier *face_cascade);
bool getInitialFitting(Mat image, Rect face, std::vector<Point2f> s0,
    CascadeClassifier eyes_cascade, Mat & R, Point2f & Trans, float & scale);
bool parseArguments(int argc, char** argv, String & cascade,
    String & model, String & images, String & annotations, String & testImages
);

int main(int argc, char** argv )
{
    FacemarkAAM::Params params;
    params.scales.clear();
    params.scales.push_back(2);
    params.scales.push_back(4);
    Ptr<FacemarkAAM> facemark = FacemarkAAM::create(params);
    facemark->loadModel("../../extra/aam_model.yaml");
        
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "Video Capture Fail" << endl;
        return 1;
    }

    Mat image;
    cap >> image;

                printf("0\n");
    //Calcula a escala e coleta a quantidade de linhas para uma imagem de 640 cols

    //! [trainsformation_variables]
    float scale ;
    Point2f T;
    Mat R;
    //! [trainsformation_variables]

                printf("1\n");
    //! [base_shape]
    FacemarkAAM::Data data;
    facemark->getData(&data);
    std::vector<Point2f> s0 = data.s0;
    //! [base_shape]

                printf("2\n");
    //! [fitting]
    //fitting process
    std::vector<Rect> faces;
    //! [load_cascade_models]
    CascadeClassifier face_cascade("../../extra/haarcascade_frontalface_alt2.xml");
                printf("3\n");
    CascadeClassifier eyes_cascade("../../extra/haarcascade_eye.xml");
    //! [load_cascade_models]
                printf("4\n");
    for(;;){
        cap >> image;
    auto showSize = Size(320, ((float)320 / image.cols) * image.rows);
        resize(image, image, showSize, 0, 0, INTER_LINEAR_EXACT);
        imshow("image", image);
        waitKey(5);
        //! [detect_face]
        myDetector(image, faces, &face_cascade);
                printf("4.5\n");
        //! [detect_face]
        if(faces.size()>0){
            
            std::vector<FacemarkAAM::Config> conf;
            std::vector<Rect> faces_eyes;
            for(unsigned j=0;j<faces.size();j++){
                if(getInitialFitting(image,faces[j],s0,eyes_cascade, R,T,scale)){
                    conf.push_back(FacemarkAAM::Config(R,T,scale,(int)params.scales.size()-1));
                    faces_eyes.push_back(faces[j]);
                }
            }
            
            if(conf.size()>0){
                printf(" - face with eyes found %i", (int)conf.size());
                std::vector<std::vector<Point2f> > landmarks;
                facemark->fitConfig(image, faces_eyes, landmarks, conf);
                for(unsigned j=0;j<landmarks.size();j++){
                    drawFacemarks(image, landmarks[j],Scalar(0,255,0));
                }
            }
            imshow("fit", image);
            
        }

    } 
}

bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
    Mat gray;

    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image.getMat().clone();

    equalizeHist(gray, gray);

    std::vector<Rect> faces_;
    face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(faces_).copyTo(faces);
    return true;
}

*/

//http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html
//FacemarkKazemi https://www.csc.kth.se/~vahidk/face_ert.html
//FacemarkAAM: https://ibug.doc.ic.ac.uk/media/uploads/documents/tzimiro_pantic_iccv2013.pdf
//FacemarkLBF: http://www.jiansun.org/papers/CVPR14_FaceAlignment.pdf

#include <opencv2/opencv.hpp>
#include <iostream>

#include "utils.hpp"
#include "detectar_rosto.hpp"
#include "detectar_olhos.hpp"
#include "piscadas_facemark_lbf.hpp"
#include "piscadas_facemark_aam.hpp"
#include "piscadas_facemark_kazemi.hpp"

using namespace cv;
using namespace std;
using namespace cv::face;

Ptr<CascadeClassifier> faceDetector;
Ptr<CascadeClassifier> eyeDetector;
Ptr<Facemark> facemark;
bool fitEmTonsDeCinza;
bool requerDetecaoDosOlhos;
std::string tipo;

void processar(Mat img);

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
        facemark = initFacemarkLBF();
    }
    else if (tipo.compare("2") == 0)
    {
        //AAM
        fitEmTonsDeCinza = false;
        requerDetecaoDosOlhos = true;
        facemark = initFacemarkAAM();   
    }
    else if (tipo.compare("3") == 0)
    {
        //Kazemi
        fitEmTonsDeCinza = false;
        requerDetecaoDosOlhos = false;
        facemark = initFacemarkKazemi();
    }
    else
    {
        std::cout << "Nenhum tipo informado." << std::endl;
    }

    //Inicia detector de facial e de olhos por Haarcascade
    faceDetector = initSimpleFaceDetector();
    if (requerDetecaoDosOlhos) eyeDetector = initSimpleEyeDetector();

    //Inicia captura dos vídeos
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Video Capture Fail" << endl;
        return 1;
    }
    Mat img; cap >> img;

    //Reescala a imagem para uma largura de 320 pixels
    auto showSize = Size(480, ((float)480 / img.cols) * img.rows);

    for (;;)
    {
        //Coleta a imagem da camera
        cap >> img;

        //Escala a imagem, converte para tons de cinza
        resize(img, img, showSize, 0, 0, INTER_LINEAR_EXACT);

        processar(img);
        imshow("Origem", img);

        waitKey(5);
    }
}


Mat imagemOriginalCinza;
const auto liminarOlhoFechado = 0.25;
bool olhoEsquerdoAberto = false, olhoDireitoAberto = false, pontosDetectados = false;
int piscadas = 0;

void processar(Mat imagemOriginal)
{
    olhoDimencoes olhoEsquerdoDimencoes, olhoDireitoDimencoes;
    vector<Rect> rostosDetectados;
    vector<vector<Point2f>> pontosFaciais;
    Mat imagemComPontosFaciais;

    {
        //Converte em tons de cinza e equaliza a imagem
        //Detecção por haarcascade funcionam bem com imagens equalizadas
        cvtColor(imagemOriginal, imagemOriginalCinza, COLOR_BGR2GRAY);
        equalizeHist(imagemOriginalCinza, imagemOriginalCinza);

        //Detecta os rostos na imagem
        faceDetector->detectMultiScale(imagemOriginal, rostosDetectados);
    }

    if (rostosDetectados.size() != 0)
    {
        imagemComPontosFaciais = imagemOriginal.clone();
        for (auto &&rostoDetec : rostosDetectados)
        {
            demarcarRostoDetectado(imagemOriginal, rostoDetec);
        }

        if (requerDetecaoDosOlhos) {

        } else {
            //Detecta os pontos faciais
            pontosDetectados = facemark->fit(fitEmTonsDeCinza
                              ? imagemOriginalCinza
                              : imagemOriginal,
                          rostosDetectados, pontosFaciais);
        }

        if (pontosDetectados)
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
                if (olhoEsquerdoDimencoes.proporcao > liminarOlhoFechado)
                {
                    //Registra que olho esta aberto
                    olhoEsquerdoAberto = true;
                }
                else
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