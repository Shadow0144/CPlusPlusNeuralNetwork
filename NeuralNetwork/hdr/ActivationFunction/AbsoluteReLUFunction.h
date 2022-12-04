#pragma once

#include "ActivationFunction/ActivationFunction.h"

// Absolute Rectified Linear Unit / Absolute Value Rectified Linear Unit
class AbsoluteReLUFunction : public ActivationFunction
{
public:
	AbsoluteReLUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

private:
	xt::xarray<double> absoluteReLU(const xt::xarray<double>& z) const;
};