#include "CrossEntropyFunction.h"

#include <iostream>

using namespace std;

const double c = pow(10, -8);

double CrossEntropyFunction::getError(xt::xarray<double> predicted, xt::xarray<double> actual)
{
	const size_t N = predicted.shape()[0];
	auto errors = actual * xt::log(predicted + c);
	auto error = -xt::sum(errors) / N;
	return error();
}

xt::xarray<double> CrossEntropyFunction::getDerivativeOfError(xt::xarray<double> predicted, xt::xarray<double> actual)
{
	auto errors = (predicted - actual);
	return errors;
}