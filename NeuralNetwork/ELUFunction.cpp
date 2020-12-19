#include "ELUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

ELUFunction::ELUFunction()
{

}

double ELUFunction::activate(double z)
{
	return ELU(z);
}

double ELUFunction::ELU(double z)
{
	return ((z < 0.0) ? (alpha * (exp(z) - 1.0)) : z);
}

xt::xarray<double> ELUFunction::ELU(const xt::xarray<double>& z)
{
	auto mask = (z > 0.0);
	return ((1.0 - mask) * (alpha * (exp(z) - 1.0)) + (mask * z));
}

xt::xarray<double> ELUFunction::feedForward(const xt::xarray<double>& inputs)
{
	return ELU(inputs);
}

xt::xarray<double> ELUFunction::getGradient(const xt::xarray<double>& sigmas)
{
	return (sigmas * xt::maximum(lastOutput + alpha, 1.0));
}

double ELUFunction::getAlpha()
{
	return alpha;
}

void ELUFunction::setAlpha(double alpha)
{
	this->alpha = alpha;
}

void ELUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}