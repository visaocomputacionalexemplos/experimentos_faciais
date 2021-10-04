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
    //Pontos olhos esquerdo 36 37 38 39 40 41
    //Pontos olhos direito  42 43 44 45 46 47
    for (auto &&pontosDeUmaFace : pontosFaciais)
    {
        //Contorna olho esquerdo
        for (int x = 36; x < 41; x++)
        {
            cv::line(img, pontosDeUmaFace[x], pontosDeUmaFace[x + 1], cv::Scalar(200, 0, 0));
        }
        cv::line(img, pontosDeUmaFace[41], pontosDeUmaFace[36], cv::Scalar(200, 0, 0));

        //Contorna olho direito
        for (int x = 42; x < 47; x++)
        {
            cv::line(img, pontosDeUmaFace[x], pontosDeUmaFace[x + 1], cv::Scalar(200, 0, 0));
        }
        cv::line(img, pontosDeUmaFace[42], pontosDeUmaFace[47], cv::Scalar(200, 0, 0));
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
        cv::line(img, pontosDeUmaFace[36], pontosDeUmaFace[39], cv::Scalar(200, 0, 0));
        meioSuperior = Point2f((pontosDeUmaFace[37].x + pontosDeUmaFace[38].x) / 2, (pontosDeUmaFace[37].y + pontosDeUmaFace[38].y) / 2);
        meioInferior = Point2f((pontosDeUmaFace[40].x + pontosDeUmaFace[41].x) / 2, (pontosDeUmaFace[40].y + pontosDeUmaFace[41].y) / 2);
        cv::line(img, meioSuperior, meioInferior, cv::Scalar(200, 0, 0));

        //Linhas olho direito
        cv::line(img, pontosDeUmaFace[42], pontosDeUmaFace[45], cv::Scalar(200, 0, 0));
        meioSuperior = Point2f((pontosDeUmaFace[43].x + pontosDeUmaFace[44].x) / 2, (pontosDeUmaFace[43].y + pontosDeUmaFace[44].y) / 2);
        meioInferior = Point2f((pontosDeUmaFace[46].x + pontosDeUmaFace[47].x) / 2, (pontosDeUmaFace[46].y + pontosDeUmaFace[47].y) / 2);
        cv::line(img, meioSuperior, meioInferior, cv::Scalar(200, 0, 0));
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