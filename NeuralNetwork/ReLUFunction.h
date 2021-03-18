#pragma once

#include "ActivationFunction.h"

// Rectified Linear Unit
class ReLUFunction : public ActivationFunction
{
public:
	ReLUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas) const;
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

private:
	xt::xarray<double> reLU(const xt::xarray<double>& z) const;
};