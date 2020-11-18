#include "SELUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

SELUFunction::SELUFunction(size_t incomingUnits, size_t numUnits)
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

double SELUFunction::activate(double z)
{
	return SELU(z);
}

double SELUFunction::SELU(double z)
{
	return (SCALE * ((z < 0.0) ? (ALPHA * (exp(z) - 1.0)) : z));
}

xt::xarray<double> SELUFunction::SELU(xt::xarray<double> z)
{
	auto mask = (z > 0.0);
	return (SCALE * ((1.0 - mask) * (ALPHA * (exp(z) - 1.0)) + (mask * z)));
}

xt::xarray<double> SELUFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	return SELU(dotProductResult);
}

xt::xarray<double> SELUFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> SELUFunction::activationDerivative()
{
	auto mask = (dotProduct(lastInput) > 0.0);
	return ((mask * SCALE) + ((1.0 - mask) * lastOutput));
}

void SELUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}