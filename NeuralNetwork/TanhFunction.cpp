#include "TanhFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

TanhFunction::TanhFunction(size_t incomingUnits, size_t numUnits)
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

double TanhFunction::activate(double z)
{
	return tanh(z);
}

xt::xarray<double> TanhFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	return xt::tanh(dotProductResult);
}

xt::xarray<double> TanhFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> TanhFunction::activationDerivative()
{
	return (1.0 - xt::pow(lastOutput, 2.0));
}

void TanhFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}