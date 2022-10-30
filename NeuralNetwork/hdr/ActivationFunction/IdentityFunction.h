#pragma once

#include "ActivationFunction.h"

// Identity / Linear / Pass-through
class IdentityFunction : public ActivationFunction
{
public:
	IdentityFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;
};