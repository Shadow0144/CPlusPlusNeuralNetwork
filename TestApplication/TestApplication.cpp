#define _USE_MATH_DEFINES
#define NOMINMAX

#pragma warning(push, 0)
#include <iostream>
#include <math.h>
#include <cmath>
#include <map>
#include <windows.h> 
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
#pragma warning(pop)

#include "NeuralNetwork.h"
#include "MeanSquareErrorFunction.h"
#include "CrossEntropyErrorFunction.h"
#include "NetworkVisualizer.h"
#include "IrisDataset.h"
#include "CatDogDataset.h"

#include "Test.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//#include "Convolution2DFunction.h"
//#include "MaxPooling2DFunction.h"
//#include "FlattenFunction.h"
#include "ReLUFunction.h"
#include "SigmoidFunction.h"
//#include "SoftmaxFunction.h"

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
const int MAX_EPOCHS = 100000;
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
    std::cout << "Predicted | Actual" << endl;
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
            std::cout << "Setosa";
            break;
        case 1:
            SetConsoleTextAttribute(hStdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            std::cout << "Versicolor";
            break;
        case 2:
            SetConsoleTextAttribute(hStdout, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            std::cout << "Virginica";
            break;
        }
        if (p == q)
        {
            SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            std::cout << " == ";
            correct++;
        }
        else
        {
            SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
            std::cout << " != ";
        }
        switch (q)
        {
        case 0:
            SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
            std::cout << "Setosa";
            break;
        case 1:
            SetConsoleTextAttribute(hStdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
            std::cout << "Versicolor";
            break;
        case 2:
            SetConsoleTextAttribute(hStdout, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            std::cout << "Virginica";
            break;
        }
        std::cout << endl;
    }
    SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
    std::cout << endl;

    std::cout << "Accuracy: " << (((double)(correct)) / ((double)(N))) << endl << endl;
}

void test_signal(int layers)
{
    size_t* layerShapes;
    ActivationFunctionType* functions;

    switch (layers)
    {
        case 7:
        case 6:
            layers = 7;
            layerShapes = new size_t[layers] { 1, 3, 3, 3, 3, 3, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::None,
              ActivationFunctionType::LeakyReLU,
              ActivationFunctionType::Softplus,
              ActivationFunctionType::ReLU,
              ActivationFunctionType::Sigmoid,
              ActivationFunctionType::Tanh,
              ActivationFunctionType::None };
            break;
        case 5:
            layerShapes = new size_t[layers] { 5, 3, 3, 3, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::None,
              ActivationFunctionType::ReLU,
              ActivationFunctionType::Sigmoid,
              ActivationFunctionType::Tanh,
              ActivationFunctionType::None };
            break;
        case 4:
            /*layerShapes = new size_t[layers] { 1, 3, 3, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::Linear,
              ActivationFunctionType::PReLU,
              ActivationFunctionType::ReLU,
              ActivationFunctionType::None };
            break;*/
        case 3:
            /*layerShapes = new size_t[layers] { 1, 3, 1 };
            functions = new ActivationFunctionType[layers]
            { DenseActivationFunction::CReLU,
              DenseActivationFunction::Maxout,
              DenseActivationFunction::None };
            break;*/
        case 2:
            layerShapes = new size_t[layers] { 1, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::None,
              ActivationFunctionType::Softsign };
            break;
        case 1:
        default:
            layers = 1;
            layerShapes = new size_t[layers] { 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::CReLU };
            break;
    }

    ErrorFunction* errorFunction = new MeanSquareErrorFunction();
    NeuralNetwork network = NeuralNetwork(true);
    network.setErrorFunction(ErrorFunctionType::MeanSquaredError);
    network.setBatchSize(20);
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
    const double RESCALE = 1.0 / 10.0;

    double twoPi = (2.0 * M_PI);
    double inc = 2.0 * twoPi / SAMPLES;
    int i = 0;
    xt::xarray<int>::shape_type shape_x = { SAMPLES, 1 };
    xt::xarray<double> training_x = xt::xarray<double>(shape_x);
    xt::xarray<int>::shape_type shape_y = { SAMPLES, 1 };
    //xt::xarray<int>::shape_type shape_y = { SAMPLES, 1, 2 };
    xt::xarray<double> training_y = xt::xarray<double>(shape_y);
    for (double t = -twoPi; t < twoPi; t += inc)
    {
        training_x(i, 0) = t * RESCALE;
        //training_y(i, 0) = t * RESCALE;
        //training_y(i, 0) = (0.3 * t + 0.5) * RESCALE;
        training_y(i, 0) = tanh(3.0 * sin(3.0 * t + 0.5)) * RESCALE;
        //training_y(i, 0) = (tanh(3.0 * sin(3.0 * t + 0.5)) + (0.5 * t)) * RESCALE;
        //training_y(i, 0) = (1.0 / (1.0 + exp(-t))) * RESCALE;
        //training_y(i, 0, 0) = cosh(3.0 * sin(3.0 * t + 0.5)) * RESCALE;
        //training_y(i, 0, 1) = cosh(3.0 * sin(3.0 * t + 0.5)) * RESCALE;
        //training_y(i, 0, 0) = -(3.0 * t + 0.5) * RESCALE;
        //training_y(i, 0, 1) = +(3.0 * t + 0.5) * RESCALE;
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
    //network.backPropagate(training_x, training_y);

    network.train(training_x, training_y, MAX_EPOCHS);

    xt::xarray<double> predicted = network.feedForward(training_x);
    std::cout << endl;
    for (int i = 0; i < SAMPLES; i += 10)
    {
        //std::cout << "Predicted: " << predicted(i, 0, 0) << " , " << predicted(i, 1, 0) << " actual: " << training_y(i, 0, 0) << " , " << training_y(i, 1, 0) << endl;
        std::cout << "Predicted: " << predicted(i, 0) << " actual: " << training_y(i, 0) << endl;
    }
    std::cout << endl;

    system("pause");
}

void test_iris(int layers)
{
    size_t* layerShapes;
    ActivationFunctionType* functions;

    switch (layers)
    {
        case 4:
            layerShapes = new size_t[layers] { 6, 6, 6, 3 };
            functions = new ActivationFunctionType[layers]
                {
                   ActivationFunctionType::None,
                   ActivationFunctionType::LeakyReLU,
                   ActivationFunctionType::Sigmoid,
                   ActivationFunctionType::None
                };
            break;
        case 3:
            layerShapes = new size_t[layers] { 6, 3, 3 };
            functions = new ActivationFunctionType[layers]
                {
                   ActivationFunctionType::None,
                   ActivationFunctionType::Sigmoid,
                   ActivationFunctionType::None
                };
            break;
        case 2:
            layerShapes = new size_t[layers] { 3, 3 };
            functions = new ActivationFunctionType[layers]
                {
                   ActivationFunctionType::None,
                   ActivationFunctionType::Sigmoid
                };
            break;
        case 1:
        default:
            layers = 1;
            layerShapes = new size_t[layers]{ 3 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::None };
            break;
    }

    NeuralNetwork network = NeuralNetwork();
    network.setErrorFunction(ErrorFunctionType::CrossEntropy);
    ImColor* classColors = new ImColor[3]
        { ImColor(1.0f, 0.0f, 0.0f, 1.0f),
          ImColor(0.0f, 1.0f, 0.0f, 1.0f),
          ImColor(0.0f, 0.0f, 1.0f, 1.0f) };
    network.displayClassificationEstimation(30, 5, classColors);

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

    //xt::xarray<double> irisPredictions = network.feedForward(irisFeatures);

    std::cout << "Training on Iris Dataset" << endl << endl;

    //network.setBatchSize(30);

    //print_iris_results(irisPredictions, irisLabels);

    network.train(irisFeatures, irisLabels, 1000);

    //irisPredictions = network.feedForward(irisFeatures);
    //print_iris_results(irisPredictions, irisLabels);

    system("pause");
}

void test_binary()
{
    // Set up the data
    ifstream in_file;
    in_file.open("mnist_binary.csv");
    auto data = xt::load_csv<double>(in_file);
    in_file.close();

    const int N = ((int)(data.shape()[0])); // Number of examples
    const int IMG_DIM = sqrt((int)(data.shape()[1])); // 28 x 28
    const int C = 1;

    const int CLASSES = 2;

    xt::xarray<double> labels = xt::col(data, 0);
    xt::xarray<double> features = xt::reshape_view(xt::view(data, xt::all(), xt::range(1, _)), { N, IMG_DIM, IMG_DIM, 1 });
    features /= 255.0;

    const int EXAMPLE_COUNT = 100;
    labels = xt::view(labels, xt::range(0, EXAMPLE_COUNT), xt::all());
    features = xt::view(features, xt::range(0, EXAMPLE_COUNT), xt::all(), xt::all(), xt::all());

    xt::xarray<double> oneHotLabels = xt::zeros<double>({ EXAMPLE_COUNT, CLASSES });
    for (int j = 0; j < EXAMPLE_COUNT; j++)
    {
        oneHotLabels(j, labels(j)) = 1.0;
    }
    labels = oneHotLabels;

    const int NUM_KERNELS_1 = 16;
    const int NUM_KERNELS_2 = 4;

    // Create the network
    NeuralNetwork network(true);
    network.addInputLayer({ (size_t)IMG_DIM, (size_t)IMG_DIM, C }); // 28x28x1
    network.addConvolution2DLayer(NUM_KERNELS_1, { 5, 5 }, C); // 28x28x1 -> 24x24x64
    network.addMaxPooling2DLayer({ 2, 2 }); // 24x24x64 -> 12x12x64
    network.addConvolution2DLayer(NUM_KERNELS_2, { 5, 5 }, NUM_KERNELS_1); // 12x12x16 -> 8x8x16
    network.addMaxPooling2DLayer({ 2, 2 }); // 8x8x16 -> 4x4x16
    network.addFlattenLayer(4 * 4 * NUM_KERNELS_2); // 4x4x16 -> 256
    network.addDenseLayer(ActivationFunctionType::ReLU, 32); // 256 -> 32
    network.addDenseLayer(ActivationFunctionType::ReLU, CLASSES); // 32 -> 2
    network.addSoftmaxLayer(-1);

    network.setErrorFunction(ErrorFunctionType::CrossEntropy);
    network.setBatchSize(10);

    const int COLS = 5;
    const int ROWS = min(EXAMPLE_COUNT / COLS, 30);
    ImColor* classColors = new ImColor[CLASSES]
    { ImColor(0.0f, 0.0f, 0.0f, 1.0f),
      ImColor(1.0f, 1.0f, 1.0f, 1.0f) };
    network.displayClassificationEstimation(ROWS, COLS, classColors);

    network.train(features, labels, MAX_EPOCHS);
}

void test_mnist()
{
    // Set up the data
    ifstream in_file;
    //in_file.open("mnist_mini.csv");
    in_file.open("mnist_fashion_test.csv");
    auto data = xt::load_csv<double>(in_file);
    in_file.close();

    const int N = ((int)(data.shape()[0])); // Number of examples
    const int IMG_DIM = sqrt((int)(data.shape()[1])); // 28 x 28
    const int C = 1;

    const int CLASSES = 10;

    xt::xarray<double> labels = xt::col(data, 0);
    xt::xarray<double> features = xt::reshape_view(xt::view(data, xt::all(), xt::range(1, _)), { N, IMG_DIM, IMG_DIM, 1 });
    features /= 255.0;

    const int EXAMPLE_COUNT = 100;//labels.shape()[0];
    labels = xt::view(labels, xt::range(0, EXAMPLE_COUNT), xt::all());
    features = xt::view(features, xt::range(0, EXAMPLE_COUNT), xt::all(), xt::all(), xt::all());

    xt::xarray<double> oneHotLabels = xt::zeros<double>({ EXAMPLE_COUNT, CLASSES });
    for (int j = 0; j < EXAMPLE_COUNT; j++)
    {
        oneHotLabels(j, labels(j)) = 1.0;
    }
    labels = oneHotLabels;

    const int NUM_KERNELS_1 = 64;
    const int NUM_KERNELS_2 = 16;

    // Create the network
    NeuralNetwork network(true);
    network.addInputLayer({ (size_t)IMG_DIM, (size_t)IMG_DIM, C }); // 28x28x1
    network.addConvolution2DLayer(NUM_KERNELS_1, { 5, 5 }, C); // 28x28x1 -> 24x24x64                                                   
    network.addMaxPooling2DLayer({ 2, 2 }); // 24x24x64 -> 12x12x64
    network.addConvolution2DLayer(NUM_KERNELS_2, { 5, 5 }, NUM_KERNELS_1); // 12x12x16 -> 8x8x16                                                         
    network.addMaxPooling2DLayer({ 2, 2 }); // 8x8x16 -> 4x4x16
    network.addFlattenLayer(4 * 4 * NUM_KERNELS_2); // 4x4x16 -> 256
    network.addDenseLayer(ActivationFunctionType::Sigmoid, 32); // 256 -> 32
    network.addDenseLayer(ActivationFunctionType::Softplus, CLASSES); // 32 -> 10
    network.addSoftmaxLayer(-1);

    network.setErrorFunction(ErrorFunctionType::CrossEntropy);
    network.setBatchSize(20);
    network.setOutputRate(1);

    const int COLS = 5;
    const int ROWS = min(EXAMPLE_COUNT / COLS, 30);
    ImColor* classColors = new ImColor[CLASSES]
    { ImColor(0.0f, 0.0f, 0.0f, 1.0f),
      ImColor(0.333f, 0.0f, 0.0f, 1.0f),
      ImColor(0.667f, 0.0f, 0.0f, 1.0f),
      ImColor(1.0f, 0.0f, 0.0f, 1.0f),
      ImColor(0.0f, 0.333f, 0.0f, 1.0f),
      ImColor(0.0f,0.667f, 0.0f, 1.0f),
      ImColor(0.0f, 1.0f, 0.0f, 1.0f),
      ImColor(0.0f, 0.0f, 0.333f, 1.0f),
      ImColor(0.0f, 0.0f, 0.667f, 1.0f),
      ImColor(0.0f,0.0f, 1.0f, 1.0f) };
    network.displayClassificationEstimation(ROWS, COLS, classColors);

    network.train(features, labels, MAX_EPOCHS);
}

void test_catdog()
{
    xt::xarray<double> features;
    xt::xarray<double> labels;
    loadCatDogDataset(features, labels);
}

enum class network
{
    signal = 0,
    iris = 1,
    binary = 2,
    mnist = 3,
    catdog = 4
};

void test_network(network type, int layers = 0)
{
    switch (type)
    {
        case network::signal:
            test_signal(layers);
            break;
        case network::iris:
            test_iris(layers);
            break;
        case network::binary:
            test_binary();
            break;
        case network::mnist:
            test_mnist();
            break;
        case network::catdog:
            test_catdog();
            break;
        default:
            // Do nothing
            break;
    }
}

void test_layers()
{
    ifstream in_file;
    in_file.open("mnist_test.csv");
    auto data = xt::load_csv<double>(in_file);
    in_file.close();
    int exampleCount = ((int)(data.shape()[0])); // Number of examples
    int width = sqrt((int)(data.shape()[1])); // 28 x 28
    auto classes = xt::col(data, 0);
    xt::xarray<double> features = xt::reshape_view(xt::view(data, xt::all(), xt::range(1, _)), { exampleCount, width, width });
    features /= 255.0;

    //Convolution2DFunction convfunc1({ 5, 5 }, 1, 1, 16); // 28x28x1 -> 24x24x16
    //MaxPooling2DFunction poolfunc1({ 2, 2 }); // 24x24x16 -> 12x12x16
    //Convolution2DFunction convfunc2({ 5, 5 }, 16, 1, 16); // 12x12x16 -> 8x8x16
    //MaxPooling2DFunction poolfunc2({ 2, 2 }); // 8x8x16 -> 4x4x16
    //FlattenFunction flatfunc1(256); // 4x4x16 -> 256
    //ReLUFunction densefunc1(256, 32); // 256 -> 32
    //ReLUFunction densefunc2(32, 10); // 32 -> 10
    //SoftmaxFunction softfunc1(10, -1); // 10 -> 10

    xt::xarray<double> example;
    cv::Mat resultMat;
    xt::xarray<double> result;

    double correct = 0;
    const double CASES = 100;
    for (int i = 0; i < CASES; i++)
    {
        example = xt::strided_view(features, { i, xt::all(), xt::all() });
        example.reshape({ 1, example.shape()[0], example.shape()[1], 1 });

        //result = convfunc1.feedForward(example);
        //cv::imshow("C1", convertChannelsToMat3(result, 0, 0, 3));
        //result = poolfunc1.feedForward(result);
        //result = convfunc2.feedForward(result);
        //cv::imshow("C2", convertChannelsToMat3(result, 0, 0, 3));
        //result = poolfunc2.feedForward(result);
        //result = flatfunc1.feedForward(result);
        //result = densefunc1.feedForward(result);
        //result = densefunc2.feedForward(result);
        //result = softfunc1.feedForward(result);

        //std::cout << classes(i) << ": " << xt::argmax(result) << endl;
        correct += (classes(i) == xt::argmax(result)(0)) ? 1 : 0;
    }
    cout << "Accuracy: " << (correct / CASES) << endl << endl;

    size_t batchSize = 20;
    const int ITERATIONS = 20000;
    int len = classes.shape()[0];
    for (int i = 0; i < ITERATIONS; i++)
    {
        xt::xarray<double> answers = xt::zeros<double>({ (int)batchSize, 10 });
        for (int j = 0; j < batchSize; j++)
        {
            answers(j, classes((i * batchSize + j) % len)) = 1.0;
        }

        xt::xarray<double> examples = xt::strided_view(features, { xt::range((i * batchSize) % len, ((i + 1) * batchSize) % len), xt::ellipsis() });
        examples.reshape({ (int)batchSize, (int)(width), (int)(width), 1 });

        xt::xarray<double> predicted;

        //cv::imshow("Example", convertToMat(examples));
        //auto predicted = convfunc1.feedForward(examples);
        //cv::imshow("C1", convertChannelsToMat3(predicted, 0, 0, 3));
        //auto predicted = poolfunc1.feedForward(examples); //poolfunc1.feedForward(predicted);
        //predicted = convfunc2.feedForward(predicted);
        //cv::imshow("C2", convertChannelsToMat3(predicted, 0, 0, 3));
        //predicted = poolfunc2.feedForward(predicted);
        //predicted = flatfunc1.feedForward(predicted);
        //for (int i = 0; (i < predicted.shape()[1] && i < 20); i++)
        //{
        //    cout << predicted(0, i) << " ";
        //}
        //cout << endl;
        //auto predicted = densefunc1.feedForward(examples); //densefunc1.feedForward(predicted);
        //predicted = densefunc2.feedForward(predicted);
        //predicted = softfunc1.feedForward(predicted);
        //std::cout << "Waiting..." << endl;
        //cv::waitKey(1);
        //std::cout << "Continuing..." << endl;

        xt::xarray<double> back = (predicted - answers);

        /*for (int j = 0; j < batchSize; j++)
        {
            for (int k = 0; k < 10; k++)
            {
                cout << back(j, k) << " ";
            }
            cout << endl;
        }*/

        //back = softfunc1.backPropagateCrossEntropy(back);
        //back = densefunc2.backPropagate(back);
        //back = densefunc1.backPropagate(back);
        /*back = flatfunc1.backPropagate(back);
        back = poolfunc2.backPropagate(back);
        back = convfunc2.backPropagate(back);
        back = poolfunc1.backPropagate(back);
        back = convfunc1.backPropagate(back);*/

        /*cout << "Delta2: " << endl;
        xt::xarray<double> ddf2 = densefunc2.getWeights().getDeltaParameters();
        for (int i = 0; i < ddf2.shape()[0]; i++)
        {
            for (int j = 0; j < ddf2.shape()[1]; j++)
            {
                cout << ddf2(i, j) << " ";
            }
            cout << endl;
        }*/

        /*cout << "Delta1: " << endl;
        xt::xarray<double> ddf1 = densefunc1.getWeights().getDeltaParameters();
        for (int i = 0; i < ddf1.shape()[0]; i++)
        {
            for (int j = 0; j < ddf1.shape()[1]; j++)
            {
                cout << ddf1(i, j) << " ";
            }
            cout << endl;
        }*/

        //softfunc1.applyBackPropagate();
        //densefunc2.applyBackPropagate();
        //densefunc1.applyBackPropagate();
        //flatfunc1.applyBackPropagate();
        //poolfunc2.applyBackPropagate();
        //convfunc2.applyBackPropagate();
        //poolfunc1.applyBackPropagate();
        //convfunc1.applyBackPropagate();

        const int ITERATION_PRINT = 100;
        if (i % ITERATION_PRINT == (ITERATION_PRINT - 1))
        {
            std::cout << "Iteration " << (i + 1) << " complete" << endl;
        }
        else { }

        const int ACCURACY_PRINT = 100;
        if (i % ACCURACY_PRINT == (ACCURACY_PRINT-1))
        {
            correct = 0.0;
            std::cout << endl;
            for (int i = 0; i < CASES; i++)
            {
                example = xt::strided_view(features, { i, xt::all(), xt::all() });
                example.reshape({ 1, example.shape()[0], example.shape()[1], 1 });
                //result = convfunc1.feedForward(example);
                //result = poolfunc1.feedForward(result);
                //result = convfunc2.feedForward(result);
                //result = poolfunc2.feedForward(result);
                //result = flatfunc1.feedForward(result);
                //result = densefunc1.feedForward(result);
                //result = densefunc2.feedForward(result);
                //result = softfunc1.feedForward(result);
                correct += (classes(i) == xt::argmax(result)(0)) ? 1 : 0;
            }
            std::cout << "Accuracy: " << (correct / CASES) << endl << endl;
        }
        else { }

        const int SUMS_PRINT = 2000;
        if (i % SUMS_PRINT == (SUMS_PRINT-1))
        {
            for (int k = 0; k < 10; k++)
            {
                xt::xarray<double> sums = xt::zeros<double>({ 1, 10 });
                correct = 0.0;
                double count = 0.0;
                for (int j = 0; j < CASES; j++)
                {
                    if (classes(j) == k)
                    {
                        example = xt::strided_view(features, { j, xt::all(), xt::all() });
                        example.reshape({ 1, example.shape()[0], example.shape()[1], 1 });
                        //result = convfunc1.feedForward(example);
                        //result = poolfunc1.feedForward(result);
                        //result = convfunc2.feedForward(result);
                        //result = poolfunc2.feedForward(result);
                        //result = flatfunc1.feedForward(result);
                        //result = densefunc1.feedForward(result);
                        //result = densefunc2.feedForward(result);
                        //result = softfunc1.feedForward(result);
                        sums += result;
                        correct += (k == xt::argmax(result)(0)) ? 1 : 0;
                        count++;
                    }
                }
                sums /= count;
                cout << "Accuracy: " << k << ": " << (correct / count) << endl;
                cout << "Sums: " << k << ": ";
                for (int n = 0; n < 10; n++) cout << sums(0, n) << " ";
                cout << endl;
            }
            std::cout << endl;
        }
        else { }
    }

    correct = 0;
    std::cout << endl;
    for (int i = 0; i < CASES; i++)
    {
        example = xt::strided_view(features, { i, xt::all(), xt::all() });
        example.reshape({ 1, example.shape()[0], example.shape()[1], 1 });

        //result = convfunc1.feedForward(example);
        //cv::imshow("C1", convertChannelsToMat3(result, 0, 0, 3));
        //result = poolfunc1.feedForward(result);
        //result = convfunc2.feedForward(result);
        //cv::imshow("C2", convertChannelsToMat3(result, 0, 0, 3));
        //result = poolfunc2.feedForward(result);
        //result = flatfunc1.feedForward(result);
        //result = densefunc1.feedForward(result);
        //result = densefunc2.feedForward(result);
        //result = softfunc1.feedForward(result);

        //std::cout << classes(i) << ": " << xt::argmax(result) << endl;
        correct += (classes(i) == xt::argmax(result)(0)) ? 1 : 0;
    }
    cout << "Accuracy: " << (correct / CASES) << endl;

    //result = convfunc2.getWeights().getParameters();
    for (int i = 0; i < 10; i++)
    {
        resultMat = convertWeightsToMat3(result, 0, 1, i);
        //cv::imshow("Weight-"+i, resultMat);
    }
    std::cout << "Waiting..." << endl;
    //cv::waitKey();
    std::cout << "Continuing..." << endl;
    system("pause");
}

int main(int argc, char** argv)
{
    test_network(network::signal, 2);

    //test_layers();

    //test_binary();

    return 0;
}