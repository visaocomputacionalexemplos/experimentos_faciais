#include <opencv2/opencv.hpp>

#include "piscadas_facemark_lbf.hpp"

using namespace cv;
using namespace cv::face;

Ptr<Facemark> initFacemarkAAM()
{
    if (true) {
        Ptr<FacemarkAAM> facemark = FacemarkAAM::create();
        facemark->loadModel("../../extra/aam_model.yaml");
        return facemark;
    }

    FacemarkAAM::Params params;
    params.model_filename = "../../extra/aam_model.yaml"; // filename to save the trained model
    params.save_model = true;
    params.verbose = true;
    params.max_m = 550;
    params.max_n = 136;
    params.texture_max_m = 145;

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
    //facemark->save("/home/piemontez/Projects/piemontez/vc/aam/aam_model.yaml");

    return facemark;
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

    http://amroamroamro.github.io/mexopencv/opencv_contrib/facemark_aam_train_demo.html
*/