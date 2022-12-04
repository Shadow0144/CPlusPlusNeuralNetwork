#include "ActivationFunction/SwishFunction.h"
#include "NeuralLayer/NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

SwishFunction::SwishFunction()
{

}

double SwishFunction::activate(double z) const
{
	return swish(z);
}

double SwishFunction::swish(double z) const
{
	return (z * sigmoid(z));
}

xt::xarray<double> SwishFunction::swish(const xt::xarray<double>& z) const
{
	return (z * sigmoid(z));
}

xt::xarray<double> SwishFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return swish(inputs);
}

xt::xarray<double> SwishFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	auto sig = sigmoid(lastInput);
	return (sigmas * (lastOutput + (sig * (1.0 - lastOutput))));
}

void SwishFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}