#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

#include "NeuralNetwork.h"

using namespace cv;
using namespace std;

//#define FIVE
//#define FOUR
#define THREE
//#define TWO
//#define ONE

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

void draw()
{
    int w = 800;

    char window_name[] = "Neural Network";
    Mat img = Mat::Mat(w, w, CV_8UC3, Scalar(225, 225, 225));
    Point pt(w / 2, w / 2);

    circle(img, pt, w / 32, Scalar(0, 0, 255), FILLED, LINE_8);

    float start_point_x = 20;
    float end_point_x = 100;
    float range = (end_point_x - start_point_x);
    float half_range = range / 2.0f;
    vector<Point2f> curvePoints;

    //Define the curve through equation. In this example, a simple parabola
    for (float x = start_point_x; x <= end_point_x; x++) 
    {
        float y = 150.0f - 50.0f * (1.0f / (1.0f + exp(-(x - half_range))));
        Point2f new_point = Point2f(2 * x, 2 * y);
        curvePoints.push_back(new_point); 
    }

    Mat curve(curvePoints, true);
    curve.convertTo(curve, CV_32S);
    polylines(img, curve, false, Scalar(0, 0, 255));

    imshow(window_name, img);
    waitKey(0); // Wait for a keystroke in the window
}

void draw_network()
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
    int layerShapes[] = { 1, 3, 1 };
    ActivationFunction functions[] =
        { ActivationFunction::WeightedDotProduct,
        ActivationFunction::Sigmoid,
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

    NeuralNetwork network = NeuralNetwork(layers, layerShapes, functions);

    DrawingCanvas canvas;
    canvas.canvas = img;
    canvas.offset = Point(0, 0);
    canvas.scale = 1.0f;
    network.draw(canvas);

    int samples = 10;
    float x[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    //float x[1] = { 1 };
    float y[10] = { 3, 5, 7, 8, 11, 13, 15, 17, 19, 21 };
    //float y[1] = { 2 };
    Mat training_x = cv::Mat(samples, 1, CV_32F, x) / 100.0f;
    Mat training_y = cv::Mat(samples, 1, CV_32F, y) / 100.0f;

    Mat result_y = network.feedForward(training_x);

    for (int i = 0; i < training_x.rows; i++)
    {
        cout << "Feedforward Untrained: X: " << training_x.at<float>(i) << " Y': " << training_y.at<float>(i) << " Y: " << result_y.at<float>(i) << endl;
    }

    cout << endl;

    network.draw(canvas);
    imshow(window_name, img);
    moveWindow(window_name, 0, 0);

    const int EPOCHS = 1000;
    const int PRINT = 100;
    for (int t = 0; t < EPOCHS; t++)
    {
        network.backPropagate(training_x, training_y);

        if (t % PRINT == 0)
        {
            network.draw(canvas);
            imshow(window_name, img);
            waitKey(100); // Wait 100ms
            result_y = network.feedForward(training_x);
            cout << endl << "Epoch: " << t << endl;
            for (int i = 0; i < training_x.rows; i++)
            {
                cout << "Feedforward Training: X: " << training_x.at<float>(i) 
                    << " Y': " << training_y.at<float>(i) 
                    << " Y: " << result_y.at<float>(i) << endl;
            }
        }
    }
    result_y = network.feedForward(training_x);

    cout << endl;

    for (int i = 0; i < training_x.rows; i++)
    {
        cout << "Feedforward Trained: X: " << training_x.at<float>(i) << " Y': " << training_y.at<float>(i) << " Y: " << result_y.at<float>(i) << endl;
    }

    imshow(window_name, img);
    waitKey(0); // Wait for a keystroke in the window
}

int main()
{
    //std::cout << "Hello World!\n";

    draw_network();

    return 0;
}