#include "SwishFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

SwishFunction::SwishFunction(size_t incomingUnits, size_t numUnits)
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

double SwishFunction::sigmoid(double z)
{
	return (1.0 / (1.0 + exp(-z)));
}

xt::xarray<double> SwishFunction::sigmoid(xt::xarray<double> z)
{
	return (1.0 / (1.0 + exp(-z)));
}

double SwishFunction::activate(double z)
{
	return swish(z);
}

double SwishFunction::swish(double z)
{
	return (z * sigmoid(z));
}

xt::xarray<double> SwishFunction::swish(xt::xarray<double> z)
{
	lastSigmoid = sigmoid(z);
	return (z * lastSigmoid);
}

xt::xarray<double> SwishFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = swish(dotProductResult);
	return lastOutput;
}

xt::xarray<double> SwishFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> SwishFunction::activationDerivative()
{
	return (lastOutput + (lastSigmoid * (1.0 - lastOutput)));
}

void SwishFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}