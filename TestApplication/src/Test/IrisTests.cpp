#include "Test/Tests.h"
#include "TestApplication.h"
#include "Dataset/IrisDataset.h"

using namespace xt::placeholders;
using namespace std;

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

void test_iris(int layers)
{
    size_t* layerShapes;
    ActivationFunctionType* functions;

    switch (layers)
    {
        case 4:
            layerShapes = new size_t[layers]{ 6, 6, 6, 3 };
            functions = new ActivationFunctionType[layers]
            {
               ActivationFunctionType::Identity,
               ActivationFunctionType::LeakyReLU,
               ActivationFunctionType::Sigmoid,
               ActivationFunctionType::Identity
            };
            break;
        case 3:
            layerShapes = new size_t[layers]{ 6, 3, 3 };
            functions = new ActivationFunctionType[layers]
            {
               ActivationFunctionType::Identity,
               ActivationFunctionType::Sigmoid,
               ActivationFunctionType::Identity
            };
            break;
        case 2:
            layerShapes = new size_t[layers]{ 3, 3 };
            functions = new ActivationFunctionType[layers]
            {
               ActivationFunctionType::Identity,
               ActivationFunctionType::Sigmoid
            };
            break;
        case 1:
        default:
            layers = 1;
            layerShapes = new size_t[layers]{ 3 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::Identity };
            break;
    }

    NeuralNetwork network = NeuralNetwork();
    std::map<string, double> optimizerParams;
    optimizerParams[Optimizer::ETA] = 0.01;
    network.setOptimizer(OptimizerType::SGD, optimizerParams);
    network.setLossFunction(LossFunctionType::CrossEntropy);
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

    std::system("pause");
}