#include "ActivationFunction/TanhFunction.h"
#include "NeuralLayer/NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

TanhFunction::TanhFunction()
{

}

double TanhFunction::activate(double z) const
{
	return tanh(z);
}

xt::xarray<double> TanhFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return xt::tanh(inputs);
}

xt::xarray<double> TanhFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return (sigmas * (1.0 - xt::pow(lastOutput, 2.0)));
}

void TanhFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}