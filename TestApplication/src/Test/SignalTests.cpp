#include "Test/Tests.h"
#include "TestApplication.h"

using namespace xt::placeholders;
using namespace std;

void test_signal(int layers)
{
    size_t* layerShapes;
    ActivationFunctionType* functions;

    switch (layers)
    {
        case 7:
        case 6:
            layers = 7;
            layerShapes = new size_t[layers]{ 1, 3, 3, 3, 3, 3, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::Identity,
              ActivationFunctionType::LeakyReLU,
              ActivationFunctionType::Softplus,
              ActivationFunctionType::ReLU,
              ActivationFunctionType::Sigmoid,
              ActivationFunctionType::Tanh,
              ActivationFunctionType::Identity };
            break;
        case 5:
            layerShapes = new size_t[layers]{ 5, 3, 3, 3, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::Identity,
              ActivationFunctionType::ReLU,
              ActivationFunctionType::Sigmoid,
              ActivationFunctionType::Tanh,
              ActivationFunctionType::Identity };
            break;
        case 4:
            /*layerShapes = new size_t[layers] { 1, 3, 3, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::Linear,
              ActivationFunctionType::PReLU,
              ActivationFunctionType::ReLU,
              ActivationFunctionType::Identity };
            break;*/
        case 3:
            layerShapes = new size_t[layers]{ 1, 8, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::Identity,
              ActivationFunctionType::Sigmoid,
              ActivationFunctionType::Identity };
            break;
        case 2:
            layerShapes = new size_t[layers]{ 1, 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::Identity,
              ActivationFunctionType::Softsign };
            break;
        case 1:
        default:
            layers = 1;
            layerShapes = new size_t[layers]{ 1 };
            functions = new ActivationFunctionType[layers]
            { ActivationFunctionType::CReLU };
            break;
    }

    NeuralNetwork network = NeuralNetwork(true);

    vector<size_t> inputShape;
    inputShape.push_back(1);
    network.addInputLayer(inputShape);
    for (int i = 0; i < layers; i++)
    {
        network.addDenseLayer(functions[i], layerShapes[i]);
    }
    //network.addMaxoutLayer(3, 3);
    //network.addDenseLayer(ActivationFunctionType::Identity, 9);
    //network.addReshapeLayer({ 3, 3 });
    //network.addFlattenLayer(9);
    //network.addDropoutLayer();
    //network.addDenseLayer(ActivationFunctionType::ReLU, 6);
    //network.addDenseLayer(ActivationFunctionType::Sigmoid, 6);
    //network.addDenseLayer(ActivationFunctionType::Identity, 1);

    //network.enableStoppingCondition(StoppingCondition::Min_Delta_Loss, 1e-8);
    //network.setLossFunction(LossFunctionType::MeanSquaredError);
    network.setLossFunction(LossFunctionType::CrossEntropy);

    std::map<string, double> optimizerParams;
    /*optimizerParams[SGDOptimizer::GAMMA] = 0.9;
    optimizerParams[SGDOptimizer::NESTEROV] = 1.0; // Enable*/
    //optimizerParams[SGDOptimizer::ETA] = 0.002;
    //optimizerParams[FtrlOptimizer::BATCH_SIZE] = 10;
    //optimizerParams[Optimizer::LAMDA1] = 0.0001;
    //optimizerParams[Optimizer::LAMDA2] = 0.0001;
    network.setOptimizer(OptimizerType::RProp, optimizerParams);
    network.displayRegressionEstimation();

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
        training_y(i, 0) = tanh(3.0 * sin(1.0 * t + 0.5)) * RESCALE;
        //training_y(i, 0) = (tanh(3.0 * sin(3.0 * t + 0.5)) + (0.5 * t)) * RESCALE;
        //training_y(i, 0) = (1.0 / (1.0 + exp(-t))) * RESCALE;
        //training_y(i, 0, 0) = cosh(3.0 * sin(3.0 * t + 0.5)) * RESCALE;
        //training_y(i, 0, 1) = cosh(3.0 * sin(3.0 * t + 0.5)) * RESCALE;
        //training_y(i, 0, 0) = -(3.0 * t + 0.5) * RESCALE;
        //training_y(i, 0, 1) = +(3.0 * t + 0.5) * RESCALE;
        i++;
    }

    // Shuffle
    bool shuffle = true;
    if (shuffle)
    {
        xt::xstrided_slice_vector svI({ 0, xt::ellipsis() });
        xt::xstrided_slice_vector svJ({ 0, xt::ellipsis() });
        const size_t N = training_x.shape()[0];
        for (size_t i = N - 1; i > 0; i--)
        {
            size_t j = rand() % i;
            svI[0] = i;
            svJ[0] = j;
            auto x = xt::xarray<double>(xt::strided_view(training_x, svI));
            xt::strided_view(training_x, svI) = xt::strided_view(training_x, svJ);
            xt::strided_view(training_x, svJ) = x;
            auto y = xt::xarray<double>(xt::strided_view(training_y, svI));
            xt::strided_view(training_y, svI) = xt::strided_view(training_y, svJ);
            xt::strided_view(training_y, svJ) = y;
        }
    }
    else { }

    //network.feedForward(training_x);
    //network.getDeltaWeight(training_x, training_y);
    network.train(training_x, training_y, MAX_EPOCHS);

    xt::xarray<double> predicted = network.predict(training_x);
    std::cout << endl;
    for (int i = 0; i < SAMPLES; i += 10)
    {
        //std::cout << "Predicted: " << predicted(i, 0, 0) << " , " << predicted(i, 1, 0) << " actual: " << training_y(i, 0, 0) << " , " << training_y(i, 1, 0) << endl;
        std::cout << "Predicted: " << predicted(i, 0) << " actual: " << training_y(i, 0) << endl;
    }
    std::cout << endl;

    std::system("pause");
}

void test_signal_reshape()
{
    NeuralNetwork network = NeuralNetwork(true);

    vector<size_t> inputShape;
    inputShape.push_back(1);
    network.addInputLayer(inputShape);
    network.addDenseLayer(ActivationFunctionType::Identity, 9);
    network.addReshapeLayer({ 3, 3 });
    network.addFlattenLayer();
    network.addReshapeLayer({ 3, 1, 1, 3 });
    network.addSqueezeLayer({ 1 });
    network.addFlattenLayer();
    network.addReshapeLayer({ 3, 1, 1, 3, 1 });
    network.addSqueezeLayer();
    network.addFlattenLayer();
    network.addDenseLayer(ActivationFunctionType::Sigmoid, 6);
    network.addDenseLayer(ActivationFunctionType::Identity, 1);

    network.setLossFunction(LossFunctionType::MeanSquaredError);
    network.setOptimizer(OptimizerType::Adam);
    network.displayRegressionEstimation();

    const int SAMPLES = 100;
    const double RESCALE = 1.0 / 10.0;

    double twoPi = (2.0 * M_PI);
    double inc = 2.0 * twoPi / SAMPLES;
    int i = 0;
    xt::xarray<int>::shape_type shape_x = { SAMPLES, 1 };
    xt::xarray<double> training_x = xt::xarray<double>(shape_x);
    xt::xarray<int>::shape_type shape_y = { SAMPLES, 1 };
    xt::xarray<double> training_y = xt::xarray<double>(shape_y);
    for (double t = -twoPi; t < twoPi; t += inc)
    {
        training_x(i, 0) = t * RESCALE;
        training_y(i, 0) = tanh(3.0 * sin(1.0 * t + 0.5)) * RESCALE;
        i++;
    }

    // Shuffle
    bool shuffle = true;
    if (shuffle)
    {
        xt::xstrided_slice_vector svI({ 0, xt::ellipsis() });
        xt::xstrided_slice_vector svJ({ 0, xt::ellipsis() });
        const size_t N = training_x.shape()[0];
        for (size_t i = N - 1; i > 0; i--)
        {
            size_t j = rand() % i;
            svI[0] = i;
            svJ[0] = j;
            auto x = xt::xarray<double>(xt::strided_view(training_x, svI));
            xt::strided_view(training_x, svI) = xt::strided_view(training_x, svJ);
            xt::strided_view(training_x, svJ) = x;
            auto y = xt::xarray<double>(xt::strided_view(training_y, svI));
            xt::strided_view(training_y, svI) = xt::strided_view(training_y, svJ);
            xt::strided_view(training_y, svJ) = y;
        }
    }
    else { }
    network.train(training_x, training_y, MAX_EPOCHS);

    xt::xarray<double> predicted = network.predict(training_x);
    std::cout << endl;
    for (int i = 0; i < SAMPLES; i += 10)
    {
        std::cout << "Predicted: " << predicted(i, 0) << " actual: " << training_y(i, 0) << endl;
    }
    std::cout << endl;

    std::system("pause");
}

void test_signal_maxout()
{
    NeuralNetwork network = NeuralNetwork(true);

    vector<size_t> inputShape;
    inputShape.push_back(1);
    network.addInputLayer(inputShape);
    network.addMaxoutLayer(9, 9);
    network.addDenseLayer(ActivationFunctionType::Identity, 1);

    network.setLossFunction(LossFunctionType::MeanSquaredError);
    network.setOptimizer(OptimizerType::Adam);
    network.displayRegressionEstimation();

    const int SAMPLES = 100;
    const double RESCALE = 1.0 / 10.0;

    double twoPi = (2.0 * M_PI);
    double inc = 2.0 * twoPi / SAMPLES;
    int i = 0;
    xt::xarray<int>::shape_type shape_x = { SAMPLES, 1 };
    xt::xarray<double> training_x = xt::xarray<double>(shape_x);
    xt::xarray<int>::shape_type shape_y = { SAMPLES, 1 };
    xt::xarray<double> training_y = xt::xarray<double>(shape_y);
    for (double t = -twoPi; t < twoPi; t += inc)
    {
        training_x(i, 0) = t * RESCALE;
        training_y(i, 0) = tanh(3.0 * sin(1.0 * t + 0.5)) * RESCALE;
        i++;
    }

    // Shuffle
    bool shuffle = true;
    if (shuffle)
    {
        xt::xstrided_slice_vector svI({ 0, xt::ellipsis() });
        xt::xstrided_slice_vector svJ({ 0, xt::ellipsis() });
        const size_t N = training_x.shape()[0];
        for (size_t i = N - 1; i > 0; i--)
        {
            size_t j = rand() % i;
            svI[0] = i;
            svJ[0] = j;
            auto x = xt::xarray<double>(xt::strided_view(training_x, svI));
            xt::strided_view(training_x, svI) = xt::strided_view(training_x, svJ);
            xt::strided_view(training_x, svJ) = x;
            auto y = xt::xarray<double>(xt::strided_view(training_y, svI));
            xt::strided_view(training_y, svI) = xt::strided_view(training_y, svJ);
            xt::strided_view(training_y, svJ) = y;
        }
    }
    else { }
    network.train(training_x, training_y, MAX_EPOCHS);

    xt::xarray<double> predicted = network.predict(training_x);
    std::cout << endl;
    for (int i = 0; i < SAMPLES; i += 10)
    {
        std::cout << "Predicted: " << predicted(i, 0) << " actual: " << training_y(i, 0) << endl;
    }
    std::cout << endl;

    std::system("pause");
}

void test_signal_crelu()
{
    NeuralNetwork network = NeuralNetwork(true);

    vector<size_t> inputShape;
    inputShape.push_back(1);
    network.addInputLayer(inputShape);
    network.addDenseLayer(ActivationFunctionType::CReLU, 2);
    network.addAveragePooling1DLayer({ 2 }, { }, false);
    network.addSqueezeLayer();
    network.addDenseLayer(ActivationFunctionType::Identity, 1);

    network.setLossFunction(LossFunctionType::MeanSquaredError);
    network.setOptimizer(OptimizerType::Adam);
    network.displayRegressionEstimation();

    const int SAMPLES = 100;
    const double RESCALE = 1.0 / 10.0;

    double twoPi = (2.0 * M_PI);
    double inc = 2.0 * twoPi / SAMPLES;
    int i = 0;
    xt::xarray<int>::shape_type shape_x = { SAMPLES, 1 };
    xt::xarray<double> training_x = xt::xarray<double>(shape_x);
    xt::xarray<int>::shape_type shape_y = { SAMPLES, 1 };
    xt::xarray<double> training_y = xt::xarray<double>(shape_y);
    for (double t = -twoPi; t < twoPi; t += inc)
    {
        training_x(i, 0) = t * RESCALE;
        training_y(i, 0) = tanh(3.0 * sin(1.0 * t + 0.5)) * RESCALE;
        i++;
    }

    // Shuffle
    bool shuffle = true;
    if (shuffle)
    {
        xt::xstrided_slice_vector svI({ 0, xt::ellipsis() });
        xt::xstrided_slice_vector svJ({ 0, xt::ellipsis() });
        const size_t N = training_x.shape()[0];
        for (size_t i = N - 1; i > 0; i--)
        {
            size_t j = rand() % i;
            svI[0] = i;
            svJ[0] = j;
            auto x = xt::xarray<double>(xt::strided_view(training_x, svI));
            xt::strided_view(training_x, svI) = xt::strided_view(training_x, svJ);
            xt::strided_view(training_x, svJ) = x;
            auto y = xt::xarray<double>(xt::strided_view(training_y, svI));
            xt::strided_view(training_y, svI) = xt::strided_view(training_y, svJ);
            xt::strided_view(training_y, svJ) = y;
        }
    }
    else { }
    network.train(training_x, training_y, MAX_EPOCHS);

    xt::xarray<double> predicted = network.predict(training_x);
    std::cout << endl;
    for (int i = 0; i < SAMPLES; i += 10)
    {
        std::cout << "Predicted: " << predicted(i, 0) << " actual: " << training_y(i, 0) << endl;
    }
    std::cout << endl;

    std::system("pause");
}