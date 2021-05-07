#include "Tests.h"
#include "TestApplication.h"
#include "CatDogDataset.h"

using namespace xt::placeholders;
using namespace std;

void test_catdog()
{
    xt::xarray<double> features;
    xt::xarray<double> labels;
    loadCatDogDataset(features, labels);

    const int N = (int)features.shape()[0]; // Number of examples
    const int IMG_ROWS = (int)features.shape()[1]; // 1024
    const int IMG_COLS = (int)features.shape()[2]; // 1024
    const int CHANNELS = (int)features.shape()[3]; // 3
    const int CLASSES = (int)labels.shape()[1];

    const int NUM_KERNELS_1 = 16;
    const int NUM_KERNELS_2 = 16;
    const int NUM_KERNELS_3 = 16;
    const int NUM_KERNELS_4 = 4;

    // Create the network
    NeuralNetwork network(true);
    network.addInputLayer({ (size_t)IMG_ROWS, (size_t)IMG_COLS, (size_t)CHANNELS }); // 1024x1024x3

    network.addConvolution2DLayer(NUM_KERNELS_1, { 5 }, { 2 }, false, true, ActivationFunctionType::ReLU); // 1024x1024x3 -> 1020x1020x16                                            
    network.addMaxPooling2DLayer({ 4 }); // 1020x1020x16 -> 255x255x16

    network.addConvolution2DLayer(NUM_KERNELS_2, { 4 }, { 4 }); // 255x255x16 -> 252x252x16                                                         
    network.addMaxPooling2DLayer({ 4 }); // 506x506x16 -> 63x63x16

    network.addConvolution2DLayer(NUM_KERNELS_3, { 4 }); // 63x63x16 -> 60x60x16                                                         
    network.addMaxPooling2DLayer({ 4 }); // 60x60x16 -> 15x15x16

    network.addConvolution2DLayer(NUM_KERNELS_4, { 6 }); // 15x15x16 -> 10x10x16                                                         
    network.addMaxPooling2DLayer({ 2 }); // 10x10x16 -> 5x5x4

    network.addFlattenLayer(); // 5x5x4 -> 100
    network.addDenseLayer(ActivationFunctionType::Sigmoid, 32); // 100 -> 32
    network.addDenseLayer(ActivationFunctionType::Softplus, CLASSES); // 32 -> 2
    network.addSoftmaxLayer(-1);

    std::map<string, double> optimizerParams;
    optimizerParams[Optimizer::ETA] = 0.01;
    optimizerParams[Optimizer::BATCH_SIZE] = 1;
    network.setOptimizer(OptimizerType::SGD, optimizerParams);
    network.setLossFunction(LossFunctionType::CrossEntropy);
    network.setOutputRate(1);

    const int COLS = 5;
    const int ROWS = max(1, min(N / COLS, 30));
    ImColor* classColors = new ImColor[CLASSES]
    { ImColor(0.0f, 0.0f, 0.0f, 1.0f),
      ImColor(1.0, 1.0f, 1.0f, 1.0f) };
    network.displayClassificationEstimation(ROWS, COLS, classColors);

    std::cout << "Training on Cat / Dog Dataset" << endl << endl;

    string networkParametersFileName = "catdog";
    std::filesystem::remove_all(getCurrentFolder().append("\\" + networkParametersFileName)); ////
    network.enableAutosave(networkParametersFileName, 1);
    //network.saveParameters(networkParametersFileName);
    network.loadParameters(networkParametersFileName);

    network.train(features, labels, MAX_EPOCHS);

    network.saveParameters(networkParametersFileName);

    std::system("pause");
}