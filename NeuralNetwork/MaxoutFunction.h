#pragma once

#include "ActivationFunction.h"

// TODO!!!
// Maxout
class MaxoutFunction : public ActivationFunction
{
public:
	MaxoutFunction(size_t incomingUnits, size_t numUnits, size_t numFunctions);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	xt::xarray<double> activationDerivative();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	size_t numFunctions;
};