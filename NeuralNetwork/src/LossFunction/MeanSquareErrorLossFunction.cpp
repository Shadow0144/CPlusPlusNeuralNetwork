#include "LossFunction/MeanSquareErrorLossFunction.h"

#include "NeuralNetwork.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

double MeanSquareErrorLossFunction::getLoss(const NeuralNetwork* network, const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const
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
	if (lambda1 != 0.0 || lambda2 != 0.0)
	{
		error += network->getRegularizationLoss(lambda1, lambda2);
	}
	else { }
	return error();
}

xt::xarray<double> MeanSquareErrorLossFunction::getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const
{
	return (2.0 * (predicted - actual));
}