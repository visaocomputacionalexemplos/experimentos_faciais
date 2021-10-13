#include <math.h>
#include "utils.hpp"

using namespace cv;
using namespace std;

/**
 * Coleta o tamanho dos olhos, em quantidade de pixeis,
 * e identifica a proporção do olho aberto
 */
std::tuple<olhoDimencoes, olhoDimencoes> coletarDimensoesOlhos(const std::vector<cv::Point2f> &pontosDeUmaFace)
{
    //Pontos olhos esquerdo 36 37 38 39 40 41
    //Pontos olhos direito  42 43 44 45 46 47

    olhoDimencoes olhoEsquerdoDimencoes, olhoDireitoDimencoes;
    cv::Point2f meioSuperior, meioInferior;

    //Dimencoes olho esquerdo
    //Calcula largura olho esquerdo
    olhoEsquerdoDimencoes.largura = std::sqrt(
        std::pow(pontosDeUmaFace[39].x - pontosDeUmaFace[36].x, 2) +
        std::pow(pontosDeUmaFace[39].y - pontosDeUmaFace[36].y, 2));

    //Calcula altura olho esquerdo
    meioSuperior = Point2f((pontosDeUmaFace[37].x + pontosDeUmaFace[38].x) / 2, (pontosDeUmaFace[37].y + pontosDeUmaFace[38].y) / 2);
    meioInferior = Point2f((pontosDeUmaFace[40].x + pontosDeUmaFace[41].x) / 2, (pontosDeUmaFace[40].y + pontosDeUmaFace[41].y) / 2);
    olhoEsquerdoDimencoes.altura = std::sqrt(
        std::pow(meioSuperior.x - meioInferior.x, 2) +
        std::pow(meioSuperior.y - meioInferior.y, 2));
    //Proporção olho esquerdo
    olhoEsquerdoDimencoes.proporcao = (float)olhoEsquerdoDimencoes.altura / olhoEsquerdoDimencoes.largura;

    //Proporção olho direito
    //Calcula largura olho direito
    olhoDireitoDimencoes.largura = std::sqrt(
        std::pow(pontosDeUmaFace[42].x - pontosDeUmaFace[45].x, 2) +
        std::pow(pontosDeUmaFace[42].y - pontosDeUmaFace[45].y, 2));

    //Calcula altura olho direito
    meioSuperior = Point2f((pontosDeUmaFace[43].x + pontosDeUmaFace[44].x) / 2, (pontosDeUmaFace[43].y + pontosDeUmaFace[44].y) / 2);
    meioInferior = Point2f((pontosDeUmaFace[46].x + pontosDeUmaFace[47].x) / 2, (pontosDeUmaFace[46].y + pontosDeUmaFace[47].y) / 2);
    olhoDireitoDimencoes.altura = std::sqrt(
        std::pow(meioSuperior.x - meioInferior.x, 2) +
        std::pow(meioSuperior.y - meioInferior.y, 2));
    //Proporção olho direito
    olhoDireitoDimencoes.proporcao = (float)olhoDireitoDimencoes.altura / olhoDireitoDimencoes.largura;

    return std::make_tuple(olhoEsquerdoDimencoes, olhoDireitoDimencoes);
}

/*
* Deteção de pontos facials pelo algoritmo AAM
* Necessita da matriz de rotação facial, translação e a escala do rosto
*/
bool getInitialFitting(Mat image, Rect face, std::vector<Point2f> s0, Ptr<CascadeClassifier> eyes_cascade, Mat &R, Point2f &Trans, float &scale)
{
    std::vector<Point2f> mybase;
    std::vector<Point2f> T;
    std::vector<Point2f> base = Mat(Mat(s0) + Scalar(image.cols / 2, image.rows / 2)).reshape(2);

    std::vector<Point2f> base_shape, base_shape2;
    Point2f e1 = Point2f((float)((base[39].x + base[36].x) / 2.0), (float)((base[39].y + base[36].y) / 2.0)); //eye1
    Point2f e2 = Point2f((float)((base[45].x + base[42].x) / 2.0), (float)((base[45].y + base[42].y) / 2.0)); //eye2

    if (face.width == 0 || face.height == 0)
        return false;

    std::vector<Point2f> eye;
    bool found = false;

    Mat faceROI = image(face);
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade->detectMultiScale(faceROI, eyes, 1.1, 2, CASCADE_SCALE_IMAGE, Size(20, 20));
    if (eyes.size() == 2)
    {
        found = true;
        int j = 0;
        Point2f c1((float)(face.x + eyes[j].x + eyes[j].width * 0.5), (float)(face.y + eyes[j].y + eyes[j].height * 0.5));
        j = 1;
        Point2f c2((float)(face.x + eyes[j].x + eyes[j].width * 0.5), (float)(face.y + eyes[j].y + eyes[j].height * 0.5));

        Point2f pivot;
        double a0, a1;
        if (c1.x < c2.x)
        {
            pivot = c1;
            a0 = atan2(c2.y - c1.y, c2.x - c1.x);
        }
        else
        {
            pivot = c2;
            a0 = atan2(c1.y - c2.y, c1.x - c2.x);
        }

        // scale between the two line length in detected and base shape
        scale = (float)(norm(Mat(c1) - Mat(c2)) / norm(Mat(e1) - Mat(e2)));

        //% eyes centers in scaled base shape (not shifted)
        mybase = Mat(Mat(s0) * scale).reshape(2);
        Point2f ey1 = Point2f((float)((mybase[39].x + mybase[36].x) / 2.0), (float)((mybase[39].y + mybase[36].y) / 2.0));
        Point2f ey2 = Point2f((float)((mybase[45].x + mybase[42].x) / 2.0), (float)((mybase[45].y + mybase[42].y) / 2.0));

#define TO_DEGREE 180.0 / 3.14159265
        a1 = atan2(ey2.y - ey1.y, ey2.x - ey1.x);
        Mat rot = getRotationMatrix2D(Point2f(0, 0), (a1 - a0) * TO_DEGREE, 1.0);

        rot(Rect(0, 0, 2, 2)).convertTo(R, CV_32F);

        base_shape = Mat(Mat(R * scale * Mat(Mat(s0).reshape(1)).t()).t()).reshape(2);
        ey1 = Point2f((float)((base_shape[39].x + base_shape[36].x) / 2.0), (float)((base_shape[39].y + base_shape[36].y) / 2.0));
        ey2 = Point2f((float)((base_shape[45].x + base_shape[42].x) / 2.0), (float)((base_shape[45].y + base_shape[42].y) / 2.0));

        T.push_back(Point2f(pivot.x - ey1.x, pivot.y - ey1.y));
        Trans = Point2f(pivot.x - ey1.x, pivot.y - ey1.y);
        return true;
    }
    else
    {
        Trans = Point2f((float)(face.x + face.width * 0.5), (float)(face.y + face.height * 0.5));
    }
    return found;
}

void demarcarRostoDetectado(Mat img, const Rect &regiao)
{
    cv::rectangle(img, regiao, Scalar(10, 50, 250));
}

void demarcarPontosFaciais(Mat img, const vector<Rect> &rostosDetectados, const vector<vector<Point2f>> &pontosFaciais)
{
    for (unsigned long i = 0; i < rostosDetectados.size(); i++)
    {
        for (auto &&ponto : pontosFaciais[i])
        {
            cv::circle(img, ponto, 2, cv::Scalar(50, 0, 250), FILLED);
        }
    }
}

void demarcarContornoOlhos(Mat img, const vector<vector<Point2f>> &pontosFaciais)
{
    //Pontos olhos esquerdo 36 37 38 39 40 41
    //Pontos olhos direito  42 43 44 45 46 47
    for (auto &&pontosDeUmaFace : pontosFaciais)
    {
        //Contorna olho esquerdo
        for (int x = 36; x < 41; x++)
        {
            cv::line(img, pontosDeUmaFace[x], pontosDeUmaFace[x + 1], cv::Scalar(10, 50, 250));
        }
        cv::line(img, pontosDeUmaFace[41], pontosDeUmaFace[36], cv::Scalar(10, 50, 250));

        //Contorna olho direito
        for (int x = 42; x < 47; x++)
        {
            cv::line(img, pontosDeUmaFace[x], pontosDeUmaFace[x + 1], cv::Scalar(10, 50, 250));
        }
        cv::line(img, pontosDeUmaFace[42], pontosDeUmaFace[47], cv::Scalar(10, 50, 250));
    }
}
void tracejarRegiaoInteresse(Mat img, const vector<vector<Point2f>> &pontosFaciais)
{
    Point2f meioSuperior, meioInferior;
    //Pontos olhos esquerdo 36 37 38 39 40 41
    //Pontos olhos direito  42 43 44 45 46 47
    for (auto &&pontosDeUmaFace : pontosFaciais)
    {
        //Linhas olho esquerdo
        cv::line(img, pontosDeUmaFace[36], pontosDeUmaFace[39], cv::Scalar(50, 0, 250));
        meioSuperior = Point2f((pontosDeUmaFace[37].x + pontosDeUmaFace[38].x) / 2, (pontosDeUmaFace[37].y + pontosDeUmaFace[38].y) / 2);
        meioInferior = Point2f((pontosDeUmaFace[40].x + pontosDeUmaFace[41].x) / 2, (pontosDeUmaFace[40].y + pontosDeUmaFace[41].y) / 2);
        cv::line(img, meioSuperior, meioInferior, cv::Scalar(50, 0, 250));

        //Linhas olho direito
        cv::line(img, pontosDeUmaFace[42], pontosDeUmaFace[45], cv::Scalar(50, 0, 250));
        meioSuperior = Point2f((pontosDeUmaFace[43].x + pontosDeUmaFace[44].x) / 2, (pontosDeUmaFace[43].y + pontosDeUmaFace[44].y) / 2);
        meioInferior = Point2f((pontosDeUmaFace[46].x + pontosDeUmaFace[47].x) / 2, (pontosDeUmaFace[46].y + pontosDeUmaFace[47].y) / 2);
        cv::line(img, meioSuperior, meioInferior, cv::Scalar(50, 0, 250));
    }
}

void escreverDimensoesOlhos(cv::Mat img, const cv::Rect &rostoDetectado, const olhoDimencoes &olhoEsquerdo, const olhoDimencoes &olhoDireito)
{
    std::stringstream texto;
    texto << "E: " << std::setprecision(2) << olhoEsquerdo.proporcao;

    cv::putText(img,
                texto.str(),
                cv::Point(rostoDetectado.x, rostoDetectado.y - 10),
                cv::FONT_HERSHEY_DUPLEX, 0.7,
                cv::Scalar(0, 0, 0), 1);

    texto.clear();
    texto.str("");
    texto << " D:" << std::setprecision(2) << olhoDireito.proporcao;
    cv::putText(img,
                texto.str(),
                cv::Point(rostoDetectado.x + 100, rostoDetectado.y - 10),
                cv::FONT_HERSHEY_DUPLEX, 0.7,
                cv::Scalar(0, 0, 0), 1);
}

void escreverQtdPiscadas(cv::Mat img, const int &piscadas)
{
    std::stringstream texto;
    texto << "Piscadas: " << piscadas;

    cv::putText(img,
                texto.str(),
                cv::Point(10, 20),
                cv::FONT_HERSHEY_DUPLEX, 0.7,
                cv::Scalar(0, 0, 0), 1);
}