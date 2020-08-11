#include "MSEFunction.h"

#include <iostream>

using namespace std;

double MSEFunction::getError(xt::xarray<double> predicted, xt::xarray<double> actual)
{
	size_t n = actual.shape()[0];
	auto errors = predicted - actual;
	auto sq_errors = xt::square(errors);
	double error = xt::sum(sq_errors)();
	error /= n; // TODO divide by features as well
	return error;
}

xt::xarray<double> MSEFunction::getDerivativeOfError(xt::xarray<double> predicted, xt::xarray<double> actual)
{
	return 2.0 * (predicted - actual); // TODO Scale?
}