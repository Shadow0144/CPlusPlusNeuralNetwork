#include "ELUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

ELUFunction::ELUFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// incoming x current -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
}

double ELUFunction::activate(double z)
{
	return ELU(z);
}

double ELUFunction::ELU(double z)
{
	return ((z < 0.0) ? (alpha * (exp(z) - 1.0)) : z);
}

xt::xarray<double> ELUFunction::ELU(xt::xarray<double> z)
{
	auto mask = (z > 0.0);
	return ((1.0 - mask) * (alpha * (exp(z) - 1.0)) + (mask * z));
}

xt::xarray<double> ELUFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = ELU(dotProductResult);
	return lastOutput;
}

xt::xarray<double> ELUFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> ELUFunction::activationDerivative()
{
	return xt::maximum(lastOutput + alpha, 1.0);
}

double ELUFunction::getAlpha()
{
	return alpha;
}

void ELUFunction::setAlpha(double alpha)
{
	this->alpha = alpha;
}

void ELUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}