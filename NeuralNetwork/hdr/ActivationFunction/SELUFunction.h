#pragma once

#include "ActivationFunction/ActivationFunction.h"

// Scaled Exponential Linear Unit
class SELUFunction : public ActivationFunction
{
public:
	SELUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

private:
	double activate(double z) const;

	double SELU(double z) const;
	xt::xarray<double> SELU(const xt::xarray<double>& z) const;

	const double ALPHA = 1.67326324;
	const double SCALE = 1.05070098;
};