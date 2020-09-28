#include "SoftplusFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

SoftplusFunction::SoftplusFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// input x output -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
}

double SoftplusFunction::activate(double z)
{
	return softplus(z);
}

double SoftplusFunction::softplus(double z)
{
	return (log(1.0 + exp(k * z)) / k);
}

xt::xarray<double> SoftplusFunction::softplus(xt::xarray<double> z)
{
	return (log(1.0 + exp(k * z)) / k);
}

xt::xarray<double> SoftplusFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	return softplus(dotProductResult);
}

xt::xarray<double> SoftplusFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> SoftplusFunction::activationDerivative()
{
	return (1.0 / (1.0 + exp(-k * xt::linalg::tensordot(lastInput, weights.getParameters(), 1))));
}

double SoftplusFunction::getK() 
{
	return k;
}

void SoftplusFunction::setK(double k)
{
	this->k = k;
}

void SoftplusFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}