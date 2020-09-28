#include "Test.h"
#include "MaxPooling1DFunction.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
#include "xtensor/xindex_view.hpp"
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

using namespace std;

void test()
{
    const int n1 = 4;
    const int n2 = 4;
    const int n3 = 6;

    xt::xarray<double> input = xt::arange<double>(n1 * n2 * n3);
    input.reshape({ n1, n2, n3 });

    input(0, 0, 0) = 5;
    input(1, 1, 0) = 7;
    input(2, 0, 0) = 9;

    cout << "Input: " << input.dimension() << ", "
        << input.shape()[0] << " x " << input.shape()[1] << " x " << input.shape()[2] << endl;
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            cout << "[";
            for (int k = 0; k < n3; k++)
            {
                cout << input(i, j, k) << " ";
            }
            cout << "]";
        }
        cout << endl;
    }
    cout << endl;

    MaxPooling1DFunction func = MaxPooling1DFunction(2, 1);

    auto result = func.feedForward(input);

    cout << "Result: " << result.dimension() << ", " 
        << result.shape()[0] << " x " << result.shape()[1] << " x " << result.shape()[2] << endl;
    for (int i = 0; i < result.shape()[0]; i++)
    {
        for (int j = 0; j < result.shape()[1]; j++)
        {
            cout << "[";
            for (int k = 0; k < result.shape()[2]; k++)
            {
                cout << result(i, j, k) << " ";
            }
            cout << "]";
        }
        cout << endl;
    }
    cout << endl;
}