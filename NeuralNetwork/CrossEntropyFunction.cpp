#include "CrossEntropyFunction.h"

const double c = pow(10, -8);

double CrossEntropyFunction::getError(xt::xarray<double> predicted, xt::xarray<double> actual)
{
	size_t n = predicted.shape()[0];
	auto errors = actual * xt::log(xt::transpose(predicted) + c);
	auto error = xt::sum(errors) / n;
	return error();
}

xt::xarray<double> CrossEntropyFunction::getDerivativeOfError(xt::xarray<double> predicted, xt::xarray<double> actual)
{
	auto errors = (predicted - actual);
	return errors;
}