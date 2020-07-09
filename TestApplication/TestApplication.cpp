#define _USE_MATH_DEFINES

#include <QtGui/QGuiApplication>
#include <QtQml/QQmlApplicationEngine>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <cmath>

#include "NeuralNetwork.h"
#include "MSEFunction.h"

using namespace cv;
using namespace std;

//#define FIVE
//#define FOUR
#define THREE
//#define TWO
//#define ONE

#define VERBOSITY 0

int test()
{
    Mat image;
    image = imread("SuccessVisualStudioWindows.jpg", IMREAD_COLOR); // Read the file
    if (image.empty()) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}

void test_network()
{
    const int PRINT = 100;
    const float MIN_ERROR = 0.001f;
    const int MAX_ITERATIONS = 10000;
    const float CONVERGENCE_W = 0.001f;
    const float CONVERGENCE_E = 0.00000001f;

#if defined(FIVE)
    int layers = 5;
    int layerShapes[] = { 5, 3, 3, 3, 1 };
    ActivationFunction functions[] =
    { ActivationFunction::Identity,
      ActivationFunction::ReLU,
      ActivationFunction::Sigmoid,
      ActivationFunction::Tanh,
      ActivationFunction::WeightedDotProduct };
#elif defined(THREE)
    int layers = 3;
    int layerShapes[] = { 1, 6, 1 };
    ActivationFunction functions[] =
    { ActivationFunction::WeightedDotProduct,
      ActivationFunction::Sigmoid,
      ActivationFunction::WeightedDotProduct };
#elif defined(FOUR)
    int layers = 4;
    int layerShapes[] = { 1, 3, 3, 1 };
    ActivationFunction functions[] =
    { ActivationFunction::Identity,
      ActivationFunction::Tanh,
      ActivationFunction::ReLU,
      ActivationFunction::WeightedDotProduct };
#elif defined(TWO)
    int layers = 2;
    int layerShapes[] = { 1, 1 };
    ActivationFunction functions[] =
    { ActivationFunction::WeightedDotProduct,
      ActivationFunction::WeightedDotProduct };
#elif defined(ONE)
    int layers = 1;
    int layerShapes[] = { 1 };
    ActivationFunction functions[] =
    { ActivationFunction::WeightedDotProduct };
#endif

    ErrorFunction* errorFunction = new MSEFunction();
    NeuralNetwork network = NeuralNetwork(layers, layerShapes, functions);
    network.setTrainingParameters(errorFunction, MAX_ITERATIONS, MIN_ERROR, CONVERGENCE_E, CONVERGENCE_W);

    /* // Linear
    const int SAMPLES = 10;
    float x[SAMPLES] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    float y[SAMPLES] = { 3, 5, 7, 8, 11, 13, 15, 17, 19, 21 };
    Mat training_x = cv::Mat(SAMPLES, 1, CV_32F, x) / 10.0f;
    Mat training_y = cv::Mat(SAMPLES, 1, CV_32F, y) / 10.0f;*/

    const int SAMPLES = 100;
    float x[SAMPLES];
    float y[SAMPLES];

    float twoPi = ((float)(2.0f * M_PI));
    float inc = 2.0f * twoPi / SAMPLES;
    int i = 0;
    for (float t = -twoPi; t < twoPi; t += inc)
    {
        x[i] = t;
        y[i] = ((float)(3.0 * sin(0.5 * t + 0.5)));
        i++;
    }

    Mat training_x = cv::Mat(SAMPLES, 1, CV_32F, x) / 10.0f;
    Mat training_y = cv::Mat(SAMPLES, 1, CV_32F, y) / 10.0f;

    network.train(training_x, training_y);
}

int main(int argc, char** argv)
{
#if defined(Q_OS_WIN)
    //QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    //QGuiApplication app(argc, argv);

    //QQmlApplicationEngine engine;
    //engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    //if (engine.rootObjects().isEmpty())
    //    return -1;

    //int r = app.exec();

    test_network();

    //return r;

    return 0;
}