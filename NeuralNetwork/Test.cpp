#include "Test.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

using namespace std;

void test()
{
    std::vector<size_t> inputShape;
    inputShape.push_back(6); // n
    inputShape.push_back(3); // input
    auto inputs = 2.0 * (xt::ones<double>(inputShape) - 0.5);

    std::vector<size_t> paramShape;
    paramShape.push_back(3); // input
    paramShape.push_back(5 * 7); // features * output
    auto parameters = 2.0 * (xt::ones<double>(paramShape) - 0.5);

    auto result = xt::linalg::tensordot(inputs, parameters, 1);

    size_t features = inputShape.size() - 1;
    inputShape[features] = 5;
    inputShape.push_back(7);
    result.reshape(inputShape);

    result = xt::amax(result, { features });

    cout << "Input: n: " << inputs.shape()[0] << " i: " << inputs.shape()[1] << endl;
    cout << "Weights: i: " << parameters.shape()[0] << " f: " << parameters.shape()[1] << " o: " << parameters.shape()[2] << endl;
    cout << "D: " << result.dimension() << " [" << result.shape()[0] << ", " << result.shape()[1] << ", " << result.shape()[2] << "]" << endl;
}