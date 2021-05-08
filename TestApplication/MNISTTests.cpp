#include "Tests.h"
#include "TestApplication.h"

using namespace xt::placeholders;
using namespace std;

void test_binary_mnist()
{
    // Set up the data
    ifstream in_file;
    in_file.open("mnist_binary.csv");
    auto data = xt::load_csv<double>(in_file);
    in_file.close();

    const int N = ((int)(data.shape()[0])); // Number of examples
    const int IMG_DIM = (int)sqrt((int)(data.shape()[1])); // 28 x 28
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

    const int NUM_KERNELS_1 = 4;
    const int NUM_KERNELS_2 = 4;
    const int NUM_KERNELS_3 = 4;

    // Create the network
    NeuralNetwork network(false);

    network.addInputLayer({ (size_t)IMG_DIM, (size_t)IMG_DIM, C }); // 28x28x1
    /*network.addReshapeLayer({ 14, 14, 4, 1 });
    network.addConvolution3DLayer(NUM_KERNELS_1, { 3 }, { 2 });
    network.addMaxPooling3DLayer({ 2 });
    network.addFlattenLayer();

    network.addDenseLayer(ActivationFunctionType::ReLU, 16 * 16 * 8);
    network.addReshapeLayer({ 16, 16, 8 });*/
    network.addConvolution2DLayer(NUM_KERNELS_2, { 5 }, { 1 }, true);
    network.addMaxPooling2DLayer({ 2 });
    network.addFlattenLayer();

    network.addDenseLayer(ActivationFunctionType::ReLU, 16 * 2);
    /*network.addReshapeLayer({ 16, 2 });
    network.addConvolution1DLayer(NUM_KERNELS_3, { 5 }, { 2 });
    network.addMaxPooling1DLayer({ 2 });
    network.addFlattenLayer();*/

    network.addDenseLayer(ActivationFunctionType::Sigmoid, CLASSES);
    network.addSoftmaxLayer(-1);

    std::map<string, double> optimizerParams;
    optimizerParams[Optimizer::ETA] = 0.001;
    optimizerParams[Optimizer::BATCH_SIZE] = 10;
    network.setOptimizer(OptimizerType::Nadam, optimizerParams);
    network.setLossFunction(LossFunctionType::CrossEntropy);

    const int COLS = 5;
    const int ROWS = min(EXAMPLE_COUNT / COLS, 30);
    ImColor* classColors = new ImColor[CLASSES]
    { ImColor(0.0f, 0.0f, 0.0f, 1.0f),
      ImColor(1.0f, 1.0f, 1.0f, 1.0f) };
    network.displayClassificationEstimation(ROWS, COLS, classColors);

    network.setOutputRate(1);

    network.train(features, labels, MAX_EPOCHS);

    network.predict(xt::strided_view(features, { 0, xt::ellipsis() }));

    system("Pause");
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
    const int IMG_DIM = (int)sqrt((int)(data.shape()[1])); // 28 x 28
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
    network.addConvolution2DLayer(NUM_KERNELS_1, { 5 }); // 28x28x1 -> 24x24x64                                                   
    network.addAveragePooling2DLayer({ 2 }); // 24x24x64 -> 12x12x64
    network.addConvolution2DLayer(NUM_KERNELS_2, { 5 }); // 12x12x16 -> 8x8x16                                                         
    network.addAveragePooling2DLayer({ 2 }); // 8x8x16 -> 4x4x16
    network.addFlattenLayer(); // 4x4x16 -> 256
    network.addDenseLayer(ActivationFunctionType::ReLU, 32); // 256 -> 32
    network.addDenseLayer(ActivationFunctionType::ReLU, CLASSES); // 32 -> 10
    //network.addSoftmaxLayer(-1);

    std::map<string, double> optimizerParams;
    //optimizerParams[Optimizer::ETA] = 0.01;
    //optimizerParams[Optimizer::BATCH_SIZE] = 20;
    optimizerParams[Optimizer::LAMDA1] = 0.0001;
    optimizerParams[Optimizer::LAMDA2] = 0.0001;
    network.setOptimizer(OptimizerType::Adam, optimizerParams);
    network.setLossFunction(LossFunctionType::MeanSquaredError, 0.1);
    network.setOutputRate(1);
    //network.enableStoppingCondition(StoppingCondition::Min_Delta_Loss, 1e-5);

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