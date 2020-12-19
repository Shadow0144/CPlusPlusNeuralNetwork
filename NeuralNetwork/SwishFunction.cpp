#include "SwishFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

SwishFunction::SwishFunction()
{

}

double SwishFunction::sigmoid(double z)
{
	return (1.0 / (1.0 + exp(-z)));
}

xt::xarray<double> SwishFunction::sigmoid(const xt::xarray<double>& z)
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

xt::xarray<double> SwishFunction::swish(const xt::xarray<double>& z)
{
	return (z * sigmoid(z));
}

xt::xarray<double> SwishFunction::feedForward(const xt::xarray<double>& inputs)
{
	return swish(inputs);
}

xt::xarray<double> SwishFunction::getGradient(const xt::xarray<double>& sigmas)
{
	// TODO!!!
	//auto sig = sigmoid(dotProduct(lastInput));
	//return (lastOutput + (sig * (1.0 - lastOutput)));
	return lastInput;
}

void SwishFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}