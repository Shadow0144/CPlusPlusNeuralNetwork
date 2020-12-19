#pragma once

#include "ActivationFunction.h"

// Swish
class SwishFunction : public ActivationFunction
{
public:
	SwishFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> activationDerivative();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	double activate(double z);

	double swish(double z);
	xt::xarray<double> swish(const xt::xarray<double>& z);

	double sigmoid(double z);
	xt::xarray<double> sigmoid(const xt::xarray<double>& z);
};