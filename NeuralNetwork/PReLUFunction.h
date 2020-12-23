#pragma once

#include "ActivationFunction.h"

// TODO!!!
// Parametric Rectified Linear Unit
class PReLUFunction : public ActivationFunction
{
public:
	PReLUFunction(int numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas);
	void applyBackPropagate();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	xt::xarray<double> PReLU(const xt::xarray<double>& z);

	xt::xarray<double> a; // Leak coefficients
	xt::xarray<double> deltaA;
};