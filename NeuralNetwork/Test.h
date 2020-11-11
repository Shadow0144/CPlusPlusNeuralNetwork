#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

xt::xarray<double> test(xt::xarray<double> input);
void print_dims(xt::xarray<double> xarray);