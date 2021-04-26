#include "TestApplication.h"
#include "Tests.h"

using namespace xt::placeholders;
using namespace std;

std::string getCurrentFolder()
{
    char buffer[FILENAME_MAX];
#pragma warning(suppress : 6031)
    GetCurrentFolder(buffer, FILENAME_MAX);
    std::string currentWorkingDir(buffer);
    return currentWorkingDir;
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
            test_binary_mnist();
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
    auto data = xt::load_npy<double>(in_file);
    in_file.close();
    int exampleCount = ((int)(data.shape()[0])); // Number of examples
    int width = (int)sqrt((int)(data.shape()[1])); // 28 x 28
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
    int len = (int)classes.shape()[0];
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
        //back = densefunc2.getDeltaWeight(back);
        //back = densefunc1.getDeltaWeight(back);
        /*back = flatfunc1.getDeltaWeight(back);
        back = poolfunc2.getDeltaWeight(back);
        back = convfunc2.getDeltaWeight(back);
        back = poolfunc1.getDeltaWeight(back);
        back = convfunc1.getDeltaWeight(back);*/

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
    std::system("pause");
}

int main(int argc, char** argv)
{
    //test_network(network::signal, 3);

    //test_layers();

    //test_binary();

    test_signal_reshape();

    return 0;
}