#pragma once

#include "ActivationFunction.h"

// Gaussian Error Linear Unit
class GELUFunction : public ActivationFunction
{
public:
	GELUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> activationDerivative();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	double activate(double z);

	double GELU(double z);
	xt::xarray<double> GELU(const xt::xarray<double>& z);
};