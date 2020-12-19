#include "SigmoidFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

SigmoidFunction::SigmoidFunction()
{

}

double SigmoidFunction::activate(double z)
{
	return sigmoid(z);
}

double SigmoidFunction::sigmoid(double z)
{
	return (1.0 / (1.0 + exp(-z)));
}
	
xt::xarray<double> SigmoidFunction::sigmoid(const xt::xarray<double>& z)
{
	return (1.0 / (1.0 + exp(-z)));
}

xt::xarray<double> SigmoidFunction::feedForward(const xt::xarray<double>& inputs)
{
	return sigmoid(inputs);
}

xt::xarray<double> SigmoidFunction::activationDerivative()
{
	return lastOutput * (1.0 - lastOutput);
}

void SigmoidFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}