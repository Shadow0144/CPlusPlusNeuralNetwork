#pragma once

#include "ActivationFunction.h"

// TODO!!!
// Parametric Rectified Linear Unit
class PReLUFunction : public ActivationFunction
{
public:
	PReLUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas);
	double applyBackPropagate();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	xt::xarray<double> PReLU(const xt::xarray<double>& z);

	double a; // Leak coefficient
	double deltaA;
};