#include "LossFunction/CrossEntropyLossFunction.h"

#include "NeuralNetwork.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

CrossEntropyLossFunction::CrossEntropyLossFunction()
{
	useSoftmaxGradient = false;
}

double CrossEntropyLossFunction::getLoss(const NeuralNetwork* network, const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const
{
	const size_t N = predicted.shape()[0];
	auto errors = actual * xt::log(predicted + EPSILON);
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
	xt::xarray<double> errors;
	if (useSoftmaxGradient)
	{
		errors = getGradientSoftmax(predicted, actual);
	}
	else
	{
		errors = getGradientStandard(predicted, actual);
	}
	return errors;
}

xt::xarray<double> CrossEntropyLossFunction::getGradientStandard(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const
{
	auto errors = -(actual / (predicted + EPSILON)); // Need to account for divide-by-zero
	return errors;
}

xt::xarray<double> CrossEntropyLossFunction::getGradientSoftmax(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const
{
	auto errors = (predicted - actual);
	return errors;
}

void CrossEntropyLossFunction::checkForOptimizedGradient(NeuralLayer* finalLayer)
{
	if (finalLayer->isSoftmaxLayer())
	{
		useSoftmaxGradient = true;
		finalLayer->useSimplifiedGradient(true);
	}
	else 
	{
		useSoftmaxGradient = false;
		finalLayer->useSimplifiedGradient(false);
	}
}