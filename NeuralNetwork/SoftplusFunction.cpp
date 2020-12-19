#include "SoftplusFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

SoftplusFunction::SoftplusFunction()
{

}

double SoftplusFunction::activate(double z)
{
	return softplus(z);
}

double SoftplusFunction::softplus(double z)
{
	return (log(1.0 + exp(k * z)) / k);
}

xt::xarray<double> SoftplusFunction::softplus(const xt::xarray<double>& z)
{
	return (log(1.0 + exp(k * z)) / k);
}

xt::xarray<double> SoftplusFunction::feedForward(const xt::xarray<double>& inputs)
{
	return softplus(inputs);
}

xt::xarray<double> SoftplusFunction::activationDerivative()
{
	// TODO!!!
	//return (1.0 / (1.0 + exp(-k * xt::linalg::tensordot(lastInput, weights.getParameters(), 1))));
	return lastInput;
}

double SoftplusFunction::getK() 
{
	return k;
}

void SoftplusFunction::setK(double k)
{
	this->k = k;
}

void SoftplusFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}