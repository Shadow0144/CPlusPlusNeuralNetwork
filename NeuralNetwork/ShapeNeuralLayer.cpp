#define _USE_MATH_DEFINES

#include "ShapeNeuralLayer.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#pragma warning(pop)

xt::xarray<double> ShapeNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	lastShape = input.shape();
	return feedForward(input);
}

xt::xarray<double> ShapeNeuralLayer::backPropagate(const xt::xarray<double>& sigmas)
{
	auto sigmasPrime = xt::xarray<double>(sigmas);
	sigmasPrime.reshape(lastShape);
	return sigmasPrime;
}

double ShapeNeuralLayer::applyBackPropagate()
{
	return 0; // No parameters
}