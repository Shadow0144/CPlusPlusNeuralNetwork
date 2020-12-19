#pragma once

#include "ActivationFunction.h"

// Quadratic
class QuadraticFunction : public ActivationFunction
{
public:
	QuadraticFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
};