#include "CrossEntropyLossFunction.h"

#include "NeuralNetwork.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

const double c = pow(10, -8);

double CrossEntropyLossFunction::getLoss(const NeuralNetwork* network, const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const
{
	const size_t N = predicted.shape()[0];
	auto errors = actual * xt::log(predicted + c);
	auto error = (-xt::sum(errors) / N)();
	if (lambda1 != 0.0 || lambda2 != 0.0)
	{
		error += network->getRegularizationLoss(lambda1, lambda2);
	}
	else { }
	return error;
}

xt::xarray<double> CrossEntropyLossFunction::getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const
{
	//auto errors = -(actual / (predicted + 0.00001)); // Need to account for divide-by-zero
	auto errors = (predicted - actual);
	return errors;
}

// This is only true when combined with softmax and for one-hot vectors // TODO!!!
xt::xarray<double> CrossEntropyLossFunction::getGradientSoftmax(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const
{
	auto errors = (predicted - actual);
	return errors;
}