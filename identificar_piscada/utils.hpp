#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <tuple>

struct olhoDimencoes
{
    int largura;
    int altura;
    float proporcao;
};

std::tuple<::olhoDimencoes, ::olhoDimencoes> coletarDimensoesOlhos(const std::vector<cv::Point2f> &pontosDeUmaFace);

void demarcarRostoDetectado(cv::Mat img, const cv::Rect &regiao);
void demarcarPontosFaciais(cv::Mat img, const std::vector<cv::Rect> &rostosDetectados, const std::vector<std::vector<cv::Point2f>> &pontosFaciais);
void demarcarContornoOlhos(cv::Mat img, const std::vector<std::vector<cv::Point2f>> &pontosFaciais);
void tracejarRegiaoInteresse(cv::Mat img, const std::vector<std::vector<cv::Point2f>> &pontosFaciais);
void escreverDimensoesOlhos(cv::Mat img, const cv::Rect &rostoDetectado, const olhoDimencoes &olhoEsquerdo, const olhoDimencoes &olhoDireito);
void escreverQtdPiscadas(cv::Mat img, const int &piscadas);

bool getInitialFitting(cv::Mat image, cv::Rect face, std::vector<cv::Point2f> s0, cv::CascadeClassifier eyes_cascade, cv::Mat & R, cv::Point2f & Trans, float & scale);

#endif //UTILS_H