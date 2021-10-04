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

#endif //UTILS_H