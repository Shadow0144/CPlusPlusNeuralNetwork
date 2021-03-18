#pragma once

#include "ActivationFunction.h"

// Gaussian Error Linear Unit
class GELUFunction : public ActivationFunction
{
public:
	GELUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas) const;
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

private:
	double activate(double z) const;

	double GELU(double z) const;
	xt::xarray<double> GELU(const xt::xarray<double>& z) const;
};