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
    const int size = 600;
    char window_name[] = "Neural Network";
    Mat img = Mat::Mat(size, size, CV_8UC3, Scalar(225, 225, 225));

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

    NeuralNetwork network = NeuralNetwork(layers, layerShapes, functions, errorFunction);

    DrawingCanvas canvas;
    canvas.canvas = img;
    canvas.offset = Point(0, 0);
    canvas.scale = 1.0f;

    /*const int SAMPLES = 10;
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

    Mat result_y = network.feedForward(training_x);
    network.draw(canvas, training_x, training_y);

    cout << "Initial: " << endl;
#if (VERBOSITY == 1)
    for (int i = 0; i < training_x.rows; i++)
    {
        cout << "Feedforward Untrained: X: " << training_x.at<float>(i) << " Y': " << training_y.at<float>(i) << " Y: " << result_y.at<float>(i) << endl;
    }
#endif
    float error = network.getError(result_y, training_y);
    cout << "Error: " << error << endl;

    network.draw(canvas, training_x, training_y);
    imshow(window_name, img);
    moveWindow(window_name, 400, 180);
    waitKey(100);

    const float STOP = 0.001f;
    const int ITERATIONS = 10000;
    const int PRINT = 100;
    int t = 0;
    while (error > STOP && t < ITERATIONS)
    {
        network.backPropagate(training_x, training_y);
        error = network.getError(result_y, training_y);

        if (t % PRINT == 0)
        {
            network.draw(canvas, training_x, training_y);
            imshow(window_name, img);
            waitKey(1); // Wait enough for the window to draw
            result_y = network.feedForward(training_x);
            cout << endl << "Iterations: " << (t+1) << endl;
#if (VERBOSITY == 1)
            for (int i = 0; i < training_x.rows; i++)
            {
                cout << "Feedforward Training: X: " << training_x.at<float>(i) 
                    << " Y': " << training_y.at<float>(i) 
                    << " Y: " << result_y.at<float>(i) << endl;
            }
#endif
            cout << "Error: " << error << endl;
        }

        t++;
    }
    result_y = network.feedForward(training_x);

    if (error <= STOP)
    {
        cout << endl << "Minimum loss condition reached" << endl;
    }
    else if (t == ITERATIONS)
    {
        cout << endl << "Maximum iterations reached" << endl;
    }
    else { }

    cout << endl << "Trained: " << endl;
#if (VERBOSITY == 1)
    for (int i = 0; i < training_x.rows; i++)
    {
        cout << "Feedforward Trained: X: " << training_x.at<float>(i) << " Y': " << training_y.at<float>(i) << " Y: " << result_y.at<float>(i) << endl;
    }
#endif
    cout << "Iterations: " << t << endl;
    cout << "Error: " << network.getError(result_y, training_y) << endl;

    network.draw(canvas, training_x, training_y);
    imshow(window_name, img);
    cout << endl << "Press any key to exit" << endl;
    waitKey(0); // Wait for a keystroke in the window
}

int main(int argc, char** argv)
{
#if defined(Q_OS_WIN)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    //if (engine.rootObjects().isEmpty())
    //    return -1;

    //int r = app.exec();

    test_network();

    //return r;

    return 0;
}