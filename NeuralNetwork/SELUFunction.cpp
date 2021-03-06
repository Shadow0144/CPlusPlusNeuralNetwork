#include "SELUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

SELUFunction::SELUFunction()
{

}

double SELUFunction::activate(double z)
{
	return SELU(z);
}

double SELUFunction::SELU(double z)
{
	return (SCALE * ((z < 0.0) ? (ALPHA * (exp(z) - 1.0)) : z));
}

xt::xarray<double> SELUFunction::SELU(const xt::xarray<double>& z)
{
	auto mask = (z > 0.0);
	return (SCALE * ((1.0 - mask) * (ALPHA * (exp(z) - 1.0)) + (mask * z)));
}

xt::xarray<double> SELUFunction::feedForward(const xt::xarray<double>& inputs)
{
	return SELU(inputs);
}

xt::xarray<double> SELUFunction::getGradient(const xt::xarray<double>& sigmas)
{
	auto mask = (lastInput > 0.0);
	return (sigmas * SCALE * ((mask) + ((1.0 - mask) * ALPHA * exp(lastInput))));
}

void SELUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}