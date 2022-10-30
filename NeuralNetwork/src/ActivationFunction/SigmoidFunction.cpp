#include "SigmoidFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

SigmoidFunction::SigmoidFunction()
{

}

double SigmoidFunction::activate(double z) const
{
	return sigmoid(z);
}

xt::xarray<double> SigmoidFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return sigmoid(inputs);
}

xt::xarray<double> SigmoidFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return (sigmas * (lastOutput * (1.0 - lastOutput)));
}

void SigmoidFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}