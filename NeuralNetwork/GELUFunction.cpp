#define _USE_MATH_DEFINES

#include "GELUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <cmath>
#pragma warning(pop)

using namespace std;

const double SQRT_2_PI = sqrt(M_2_PI);

GELUFunction::GELUFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// incoming x current -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
}

double GELUFunction::GELU(double z)
{
	return (-0.5 * z * (1 + tanh(SQRT_2_PI * (z + 0.044715 * pow(z, 3)))));
}

xt::xarray<double> GELUFunction::GELU(xt::xarray<double> z)
{
	return (-0.5 * z * (1 + tanh(SQRT_2_PI * (z + 0.044715 * pow(z, 3)))));
}

double GELUFunction::activate(double z)
{
	return GELU(z);
}

xt::xarray<double> GELUFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = GELU(dotProductResult);
	return lastOutput;
}

xt::xarray<double> GELUFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> GELUFunction::activationDerivative()
{
	auto z = lastOutput;
	auto z3 = pow(z, 3);
	auto sech2 = 1.0 / cosh(0.0356774 * z3 + 0.797885 * z);
	return ((0.5 * tanh((0.0356774 * z3) + (0.797885 * z))) + (((0.0535161 * z3) + (0.398942 * z)) * sech2) + 0.5);
}

void GELUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}