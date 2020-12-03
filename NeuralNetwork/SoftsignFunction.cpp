#include "SoftsignFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

SoftsignFunction::SoftsignFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// input x output -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
}

double SoftsignFunction::activate(double z)
{
	return softsign(z);
}

double SoftsignFunction::softsign(double z)
{
	return (z / (abs(z) + 1.0));
}

xt::xarray<double> SoftsignFunction::softsign(const xt::xarray<double>& z)
{
	return (z / (abs(z) + 1.0));
}

xt::xarray<double> SoftsignFunction::feedForward(const xt::xarray<double>& inputs)
{
	auto dotProductResult = dotProduct(inputs);
	return softsign(dotProductResult);
}

xt::xarray<double> SoftsignFunction::backPropagate(const xt::xarray<double>& sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> SoftsignFunction::activationDerivative()
{
	auto z = dotProduct(lastInput);
	return (1.0 / pow((1.0 + abs(z)), 2.0));
}

void SoftsignFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}