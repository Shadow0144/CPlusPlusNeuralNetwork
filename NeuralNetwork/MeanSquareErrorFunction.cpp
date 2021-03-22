#include "MeanSquareErrorFunction.h"

#include "Optimizer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

double MeanSquareErrorFunction::getError(const xt::xarray<double>& predicted, const xt::xarray<double>& actual)
{
	size_t n = actual.shape()[0];
	auto errors = predicted - actual;
	auto sq_errors = xt::square(errors);
	xt::xarray<double> error = xt::sum(sq_errors);
	while (error.dimension() > 0)
	{
		error = xt::sum(error);
	}
	error /= n;
	return error();
}

xt::xarray<double> MeanSquareErrorFunction::getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual)
{
	return (2.0 * (predicted - actual));
}