#include "Test.h"
#include "MaxPooling1DFunction.h"
#include "MaxPooling2DFunction.h"
#include "MaxPooling3DFunction.h"
#include "AveragePooling1DFunction.h"
#include "AveragePooling2DFunction.h"
#include "AveragePooling3DFunction.h"
#include "Convolution1DFunction.h"
#include "Convolution2DFunction.h"
#include "Convolution3DFunction.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
#include "xtensor/xindex_view.hpp"
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

using namespace std;

xt::xarray<double> test(xt::xarray<double> input)
{
    const int n1 = 4;
    const int n2 = 4;
    const int n3 = 3;
    const int n4 = 6;

    //xt::xarray<double> input = xt::arange<double>(n1 * n2 * n3);
    //input.reshape({ n1, n2, n3 });

    //input(0, 0, 0) = 5;
    //input(1, 1, 0) = 7;
    //input(2, 0, 0) = 9;

    /*cout << "Input: " << input.dimension() << ", "
        << input.shape()[0] << " x " << input.shape()[1] << " x " << input.shape()[2] << endl;
    for (int i = 0; i < input.shape()[0]; i++)
    {
        for (int j = 0; j < input.shape()[1]; j++)
        {
            cout << "[";
            for (int k = 0; k < input.shape()[2]; k++)
            {
                cout << input(i, j, k) << " ";
            }
            cout << "]";
        }
        cout << endl;
    }
    cout << endl;*/

    Convolution2DFunction func({ 2, 2 }, 1, 1, 1);

    auto result = func.feedForward(input);

    /*cout << "Result: " << result.dimension() << ", "
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
    cout << endl;*/

    return result;
}

void print_dims(xt::xarray<double> xarray)
{
    const int DIMS = xarray.dimension();
    std::cout << DIMS << ", ";
    for (int i = 0; i < DIMS; i++)
    {
        cout << xarray.shape()[i] << " x ";
    }
    cout << endl;
}