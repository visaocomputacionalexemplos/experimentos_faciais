#include <opencv2/opencv.hpp>

#include "detectar_pontos_faciais_lbf.hpp"
#include "utils.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> iniciarDetectorPontosFacialAAM()
{
    Ptr<FacemarkAAM> facemark = FacemarkAAM::create();
    facemark->loadModel("../../extra/aam_model.yaml");
    return facemark;
}

bool facemarkAAMFit(FacemarkAAM *ammFacemark, Ptr<CascadeClassifier> eyeDetector, Mat imagemOriginal, std::vector<Rect> rostosDetectados, std::vector<std::vector<Point2f>> &pontosFaciais)
{
    std::vector<FacemarkAAM::Config> conf;
    std::vector<Rect> faces_fit;
    float scale;
    Point2f T;
    Mat R;
    FacemarkAAM::Data data;
    ammFacemark->getData(&data);
    std::vector<Point2f> s0 = data.s0;
    FacemarkAAM::Params params;
    params.scales.clear();
    params.scales.push_back(2);
    params.scales.push_back(4);

    for (unsigned long j = 0; j < rostosDetectados.size(); j++)
    {
        if (getInitialFitting(imagemOriginal, rostosDetectados[j], s0, eyeDetector, R, T, scale))
        {
            conf.push_back(FacemarkAAM::Config(R, T, scale, (int)params.scales.size() - 1));
            //conf.push_back(FacemarkAAM::Config(R,T,scale));
            faces_fit.push_back(rostosDetectados[j]);
        }
    }

    if (conf.size() > 0)
    {
        ammFacemark->fitConfig(imagemOriginal, faces_fit, pontosFaciais, conf);
        return true;
    }
    return false;
}

/* Training code.
    FacemarkAAM::Params params;
    params.model_filename = "../../extra/aam_model.yaml"; // filename to save the trained model
    params.save_model = true;
    params.verbose = true;
    params.scales.clear();
    params.scales.push_back(2);
    params.scales.push_back(4);

    Ptr<FacemarkAAM> facemark = FacemarkAAM::create(params);

    std::string imageFiles = "/home/piemontez/Projects/piemontez/vc/aam/images.txt";
    std::string ptsFiles = "/home/piemontez/Projects/piemontez/vc/aam/points.txt";
    std::vector<std::string> images_train;
    std::vector<std::string> landmarks_train;
    // load the list of dataset: image paths and landmark file paths
    loadDatasetList(imageFiles, ptsFiles, images_train, landmarks_train);
    Mat image;
    std::vector<Point2f> facial_points;
    for (size_t i = 0; i < images_train.size(); i++)
    {
        image = imread(images_train[i].c_str());
        loadFacePoints(landmarks_train[i], facial_points);
        facemark->addTrainingSample(image, facial_points);
    }
    facemark->training();

    http://amroamroamro.github.io/mexopencv/opencv_contrib/detectar_pontos_faciais_aam_train_demo.html
*/