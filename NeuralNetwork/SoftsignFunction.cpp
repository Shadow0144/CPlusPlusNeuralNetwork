#include "SoftsignFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

SoftsignFunction::SoftsignFunction()
{

}

double SoftsignFunction::activate(double z) const
{
	return softsign(z);
}

double SoftsignFunction::softsign(double z) const
{
	return (z / (1.0 + abs(z)));
}

xt::xarray<double> SoftsignFunction::softsign(const xt::xarray<double>& z) const
{
	return (z / (1.0 + abs(z)));
}

xt::xarray<double> SoftsignFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return softsign(inputs);
}

xt::xarray<double> SoftsignFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return (sigmas * (1.0 / pow((1.0 + abs(lastInput)), 2.0)));
}

void SoftsignFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}