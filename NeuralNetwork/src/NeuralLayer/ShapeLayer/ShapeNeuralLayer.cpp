#define _USE_MATH_DEFINES

#include "NeuralLayer/ShapeLayer/ShapeNeuralLayer.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#pragma warning(pop)

ShapeNeuralLayer::ShapeNeuralLayer(NeuralLayer* parent)
	: NeuralLayer(parent)
{
	this->numUnits = 1;
}

ShapeNeuralLayer::~ShapeNeuralLayer()
{

}

xt::xarray<double> ShapeNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	lastShape = input.shape();
	return feedForward(input);
}

xt::xarray<double> ShapeNeuralLayer::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	auto sigmasPrime = xt::xarray<double>(sigmas);
	sigmasPrime.reshape(lastShape);
	return sigmasPrime;
}

double ShapeNeuralLayer::applyBackPropagate()
{
	return 0; // No parameters
}