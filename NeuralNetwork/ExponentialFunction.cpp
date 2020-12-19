#include "ExponentialFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

ExponentialFunction::ExponentialFunction()
{

}

double ExponentialFunction::activate(double z)
{
	return exp(z);
}

xt::xarray<double> ExponentialFunction::feedForward(const xt::xarray<double>& inputs)
{
	return exp(inputs);
}

xt::xarray<double> ExponentialFunction::activationDerivative()
{
	return lastOutput;
}

void ExponentialFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}