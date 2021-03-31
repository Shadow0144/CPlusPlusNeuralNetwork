#include "CrossEntropyLossFunction.h"

#include "Optimizer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

const double c = pow(10, -8);

double CrossEntropyLossFunction::getLoss(const xt::xarray<double>& predicted, const xt::xarray<double>& actual)
{
	const size_t N = predicted.shape()[0];
	auto errors = actual * xt::log(predicted + c);
	auto error = -xt::sum(errors) / N;
	return error();
}

xt::xarray<double> CrossEntropyLossFunction::getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual)
{
	//auto errors = -(actual / (predicted + 0.00001)); // Need to account for divide-by-zero
	auto errors = (predicted - actual);
	return errors;
}

// This is only true when combined with softmax and for one-hot vectors // TODO!!!
xt::xarray<double> CrossEntropyLossFunction::getGradientSoftmax(const xt::xarray<double>& predicted, const xt::xarray<double>& actual)
{
	auto errors = (predicted - actual);
	return errors;
}