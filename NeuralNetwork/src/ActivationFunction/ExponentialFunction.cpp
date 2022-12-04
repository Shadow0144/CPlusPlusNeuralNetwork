#include "ActivationFunction/ExponentialFunction.h"
#include "NeuralLayer/NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

ExponentialFunction::ExponentialFunction()
{

}

double ExponentialFunction::activate(double z) const
{
	return exp(z);
}

xt::xarray<double> ExponentialFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return exp(inputs);
}

xt::xarray<double> ExponentialFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return (sigmas * lastOutput);
}

void ExponentialFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}