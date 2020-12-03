#include "ExponentialFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

ExponentialFunction::ExponentialFunction(size_t incomingUnits, size_t numUnits)
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

double ExponentialFunction::activate(double z)
{
	return exp(z);
}

xt::xarray<double> ExponentialFunction::feedForward(const xt::xarray<double>& inputs)
{
	auto dotProductResult = dotProduct(inputs);
	return exp(dotProductResult);
}

xt::xarray<double> ExponentialFunction::backPropagate(const xt::xarray<double>& sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> ExponentialFunction::activationDerivative()
{
	return lastOutput;
}

void ExponentialFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}