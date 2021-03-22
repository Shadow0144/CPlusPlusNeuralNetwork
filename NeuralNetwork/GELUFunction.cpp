#define _USE_MATH_DEFINES

#include "GELUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <cmath>
#pragma warning(pop)

using namespace std;

const double SQRT_2_PI = sqrt(M_2_PI);

GELUFunction::GELUFunction()
{

}

double GELUFunction::GELU(double z) const
{
	return (-0.5 * z * (1 + tanh(SQRT_2_PI * (z + 0.044715 * pow(z, 3)))));
}

xt::xarray<double> GELUFunction::GELU(const xt::xarray<double>& z) const
{
	return (-0.5 * z * (1 + tanh(SQRT_2_PI * (z + 0.044715 * pow(z, 3)))));
}

double GELUFunction::activate(double z) const
{
	return GELU(z);
}

xt::xarray<double> GELUFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return GELU(inputs);
}

xt::xarray<double> GELUFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	auto z = lastOutput;
	auto z3 = pow(z, 3);
	auto sech2 = 1.0 / cosh(0.0356774 * z3 + 0.797885 * z);
	return (sigmas * ((0.5 * tanh((0.0356774 * z3) + (0.797885 * z))) + (((0.0535161 * z3) + (0.398942 * z)) * sech2) + 0.5));
}

void GELUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}