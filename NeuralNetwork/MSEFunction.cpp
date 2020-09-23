#include "MSEFunction.h"

#include <iostream>

using namespace std;

double MSEFunction::getError(xt::xarray<double> predicted, xt::xarray<double> actual)
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

xt::xarray<double> MSEFunction::getDerivativeOfError(xt::xarray<double> predicted, xt::xarray<double> actual)
{
	return (2.0 * (predicted - actual));
}