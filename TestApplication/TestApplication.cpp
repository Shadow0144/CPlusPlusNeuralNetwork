#define _USE_MATH_DEFINES
#define NOMINMAX

#pragma warning(push, 0)
#include <iostream>
#include <math.h>
#include <cmath>
#include <windows.h> 
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#pragma warning(pop)

#include "NeuralNetwork.h"
#include "MSEFunction.h"
#include "CrossEntropyFunction.h"
#include "NetworkVisualizer.h"
#include "IrisDataset.h"

#include "Test.h"

//#define ALL
//#define FIVE
//#define FOUR
//#define THREE
//#define TWO
#define ONE
//#define SIGNAL
//#define IRIS
#define MNIST

#define VERBOSITY 0

using namespace xt::placeholders;
using namespace std;

const int PRINT = 100;
const double MIN_ERROR = 0.001f;
const int MAX_ITERATIONS = 10000;
const double CONVERGENCE_W = 0.001;
const double CONVERGENCE_E = 0.00000001;

int cv_test()
{
    /*Mat image;
    image = imread("SuccessVisualStudioWindows.jpg", IMREAD_COLOR); // Read the file
    if (image.empty()) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image); // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window*/
    return 0;
}

void print_iris_results(xt::xarray<double> predicted, xt::xarray<double> actual)
{
    HANDLE hStdout, hStdin;
    hStdin = GetStdHandle(STD_INPUT_HANDLE);
    hStdout = GetStdHandle(STD_OUTPUT_HANDLE);

    int correct = 0;
    cout << "Predicted | Actual" << endl;
    const int N = ((int)(predicted.shape()[0]));
    for (int i = 0; i < N; i++)
    {
        int p = 0;
        double pValue = predicted(i, 0);
        int q = 0;
        double qValue = actual(i, 0);
        for (int j = 1; j < 3; j++)
        {
            if (predicted(i, j) > pValue)
            {
                pValue = predicted(i, j);
                p = j;
            }
            else { }
            if (actual(i, j) > qValue)
            {
                qValue = actual(i, j);
                q = j;
            }
            else { }
        }
        switch (p)
        {
        case 0:
            SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
            cout << "Setosa";
            break;
        case 1:
            SetConsoleTextAttribute(hStdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            cout << "Versicolor";
            break;
        case 2:
            SetConsoleTextAttribute(hStdout, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            cout << "Virginica";
            break;
        }
        if (p == q)
        {
            SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            cout << " == ";
            correct++;
        }
        else
        {
            SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
            cout << " != ";
        }
        switch (q)
        {
        case 0:
            SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
            cout << "Setosa";
            break;
        case 1:
            SetConsoleTextAttribute(hStdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            cout << "Versicolor";
            break;
        case 2:
            SetConsoleTextAttribute(hStdout, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            cout << "Virginica";
            break;
        }
        cout << endl;
    }
    SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
    cout << endl;

    cout << "Accuracy: " << (((double)(correct)) / ((double)(N))) << endl << endl;
}

void test_signal(int layers)
{
    size_t* layerShapes;
    DenseActivationFunction* functions;

    switch (layers)
    {
        case 7:
        case 6:
            layers = 7;
            layerShapes = new size_t[layers] { 1, 3, 3, 3, 3, 3, 1 };
            functions = new DenseActivationFunction[layers]
            { DenseActivationFunction::Linear,
                DenseActivationFunction::LeakyReLU,
                DenseActivationFunction::Softplus,
                DenseActivationFunction::ReLU,
                DenseActivationFunction::Sigmoid,
                DenseActivationFunction::Tanh,
                DenseActivationFunction::Linear };
            break;
        case 5:
            layerShapes = new size_t[layers] { 5, 3, 3, 3, 1 };
            functions = new DenseActivationFunction[layers]
            { DenseActivationFunction::Linear,
              DenseActivationFunction::ReLU,
              DenseActivationFunction::Sigmoid,
              DenseActivationFunction::Tanh,
              DenseActivationFunction::Linear };
            break;
        case 4:
            layerShapes = new size_t[layers] { 1, 3, 3, 1 };
            functions = new DenseActivationFunction[layers]
            { DenseActivationFunction::Linear,
              DenseActivationFunction::PReLU,
              DenseActivationFunction::ReLU,
              DenseActivationFunction::Linear };
            break;
        case 3:
            layerShapes = new size_t[layers] { 1, 3, 1 };
            functions = new DenseActivationFunction[layers]
            { DenseActivationFunction::Linear,
              DenseActivationFunction::Maxout,
              DenseActivationFunction::Linear };
            break;
        case 2:
            layerShapes = new size_t[layers] { 1, 1 };
            functions = new DenseActivationFunction[layers]
            { DenseActivationFunction::Linear,
              DenseActivationFunction::Linear };
            break;
        case 1:
        default:
            layers = 1;
            layerShapes = new size_t[layers] { 1 };
            functions = new DenseActivationFunction[layers]
            { DenseActivationFunction::CReLU };
            break;
    }

    ErrorFunction* errorFunction = new MSEFunction();
    NeuralNetwork network = NeuralNetwork();
    network.setTrainingParameters(errorFunction, MAX_ITERATIONS, -MIN_ERROR, -CONVERGENCE_E, -CONVERGENCE_W); // TODO: Remove negatives
    network.setBatchSize(1);
    network.displayRegressionEstimation();

    vector<size_t> inputShape;
    inputShape.push_back(1);
    network.addInputLayer(inputShape);
    for (int i = 0; i < layers; i++)
    {
        network.addDenseLayer(functions[i], layerShapes[i]);
    }

    /* // Linear
    const int SAMPLES = 10;
    float x[SAMPLES] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    float y[SAMPLES] = { 3, 5, 7, 8, 11, 13, 15, 17, 19, 21 };
    Mat training_x = cv::Mat(SAMPLES, 1, CV_32F, x) / 10.0f;
    Mat training_y = cv::Mat(SAMPLES, 1, CV_32F, y) / 10.0f;*/
    const int SAMPLES = 100;
    const double rescale = 1.0 / 10.0;

    double twoPi = (2.0 * M_PI);
    double inc = 2.0 * twoPi / SAMPLES;
    int i = 0;
    xt::xarray<int>::shape_type shape_x = { SAMPLES, 1 };
    xt::xarray<double> training_x = xt::xarray<double>(shape_x);
    xt::xarray<int>::shape_type shape_y = { SAMPLES, 1, 2 };
    xt::xarray<double> training_y = xt::xarray<double>(shape_y);
    for (double t = -twoPi; t < twoPi; t += inc)
    {
        training_x(i, 0) = t * rescale;
        //training_y(i, 0) = t * rescale;
        //training_y(i, 0) = (0.3 * t + 0.5) * rescale;
        training_y(i, 0, 0) = tanh(3.0 * sin(0.5 * t + 0.5)) * rescale;
        training_y(i, 0, 1) = cosh(3.0 * sin(0.5 * t + 0.5)) * rescale;
        i++;
    }

    // Shuffle
    xt::xstrided_slice_vector svI({ 0, xt::ellipsis() });
    xt::xstrided_slice_vector svJ({ 0, xt::ellipsis() });
    const size_t N = training_x.shape()[0];
    for (size_t i = N - 1; i > 0; i--)
    {
        size_t j = rand() % i;
        svI[0] = i;
        svJ[0] = j;
        auto x = xt::strided_view(training_x, svI);
        xt::strided_view(training_x, svI) = xt::strided_view(training_x, svJ);
        xt::strided_view(training_x, svJ) = x;
        auto y = xt::strided_view(training_y, svI);
        xt::strided_view(training_y, svI) = xt::strided_view(training_y, svJ);
        xt::strided_view(training_y, svJ) = y;
    }

    //network.feedForward(training_x);
    network.backPropagate(training_x, training_y);

    network.train(training_x, training_y);

    xt::xarray<double> results = network.feedForward(training_x);
    cout << endl;
    for (int i = 0; i < SAMPLES; i += 10)
    {
        //cout << "Predicted: " << results(i, 0, 0) << " , " << results(i, 1, 0) << " actual: " << training_y(i, 0, 0) << " , " << training_y(i, 1, 0) << endl;
        cout << "Predicted: " << results(i, 0) << " actual: " << training_y(i, 0) << endl;
    }
    cout << endl;

    system("pause");
}

void test_iris(int layers)
{
    size_t* layerShapes;
    DenseActivationFunction* functions;

    switch (layers)
    {
        case 4:
            layerShapes = new size_t[layers] { 6, 6, 6, 3 };
            functions = new DenseActivationFunction[layers]
                {
                   DenseActivationFunction::Linear,
                   DenseActivationFunction::LeakyReLU,
                   DenseActivationFunction::Sigmoid,
                   DenseActivationFunction::Linear
                };
            break;
        case 3:
            layerShapes = new size_t[layers] { 6, 3, 3 };
            functions = new DenseActivationFunction[layers]
                {
                   DenseActivationFunction::Linear,
                   DenseActivationFunction::Sigmoid,
                   DenseActivationFunction::Linear
                };
            break;
        case 2:
            layerShapes = new size_t[layers] { 3, 3 };
            functions = new DenseActivationFunction[layers]
                {
                   DenseActivationFunction::Linear,
                   DenseActivationFunction::Sigmoid
                };
            break;
        case 1:
        default:
            layers = 1;
            layerShapes = new size_t[layers]{ 3 };
            functions = new DenseActivationFunction[layers]
            { DenseActivationFunction::Linear };
            break;
    }

    ErrorFunction* errorFunction = new CrossEntropyFunction();
    NeuralNetwork network = NeuralNetwork();
    network.setTrainingParameters(errorFunction, 100, MIN_ERROR, CONVERGENCE_E, CONVERGENCE_W);
    ImColor* classColors = new ImColor[3]
        { ImColor(1.0f, 0.0f, 0.0f, 1.0f),
            ImColor(0.0f, 1.0f, 0.0f, 1.0f),
            ImColor(0.0f, 0.0f, 1.0f, 1.0f) };
    network.setClassificationVisualizationParameters(30, 5, classColors);
    network.displayClassificationEstimation();

    IrisDataset iris;
    xt::xarray<double> irisFeatures = iris.getFeatures();
    xt::xarray<double> irisLabels = iris.getLabelsOneHot();

    vector<size_t> inputShape;
    inputShape.push_back(4);
    network.addInputLayer(inputShape);
    for (int i = 0; i < layers; i++)
    {
        network.addDenseLayer(functions[i], layerShapes[i]);
    }
    network.addSoftmaxLayer();

    xt::xarray<double> irisPredictions = network.feedForward(irisFeatures);

    cout << "Training on Iris Dataset" << endl << endl;

    //network.setBatchSize(30);

    print_iris_results(irisPredictions, irisLabels);

    network.train(irisFeatures, irisLabels);

    irisPredictions = network.feedForward(irisFeatures);
    print_iris_results(irisPredictions, irisLabels);

    system("pause");
}

void test_mnist(int layers)
{
    int* layerShapes;
    DenseActivationFunction* functions;

    switch (layers)
    {
        case 1:
            layers = 1;
            layerShapes = new int[layers] { 1 };
            functions = new DenseActivationFunction[layers]
            { DenseActivationFunction::Linear };
            break;
        default:
            layers = 0;
            layerShapes = new int[layers] {  };
            functions = new DenseActivationFunction[layers]
            {  };
            break;
    }

    ErrorFunction* errorFunction = new CrossEntropyFunction();
    NeuralNetwork network = NeuralNetwork(false);
    vector<size_t> inputShape = { 28, 28 };
    vector<size_t> convolutionShape = { 2, 2 };
    size_t stride = 1;
    network.setTrainingParameters(errorFunction, MAX_ITERATIONS, MIN_ERROR, CONVERGENCE_E, CONVERGENCE_W);

    network.addInputLayer(inputShape);
    network.addConvolution2DLayer(10, convolutionShape, stride);

    ifstream in_file;
    in_file.open("mnist_mini.csv");
    auto data = xt::load_csv<double>(in_file);
    int examples = ((int)(data.shape()[0])); // 28 x 28
    int width = ((int)(data.shape()[1])); // 28 x 28
    auto mini_classes = xt::col(data, 1);
    xt::xarray<double> mini_features = xt::reshape_view(xt::view(data, xt::all(), xt::range(1, _)), { examples, 28, 28 });

    xt::xarray<double> converted = network.feedForward(mini_features);

    //cout << xt::view(converted, 0, xt::all(), xt::all()) << endl;

    system("pause");
}

enum class network
{
    signal = 0,
    iris = 1,
    mnist = 2
};

void test_network(network type, int layers)
{
    switch (type)
    {
        case network::signal:
            test_signal(layers);
            break;
        case network::iris:
            test_iris(layers);
            break;
        case network::mnist:
            test_mnist(layers);
            break;
        default:
            // Do nothing
            break;
    }
}

int main(int argc, char** argv)
{
    test_network(network::signal, 1);

    //test();

    return 0;
}