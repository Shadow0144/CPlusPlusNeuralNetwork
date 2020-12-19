#pragma once

#include "ActivationFunction.h"

// Softplus / Smooth Rectified Linear Unit
class SoftplusFunction : public ActivationFunction
{
public:
	SoftplusFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> activationDerivative();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

	double getK();
	void setK(double k);

private:
	double activate(double z);

	double softplus(double z);
	xt::xarray<double> softplus(const xt::xarray<double>& z);

	double k = 1.0; // Sharpness coefficient
};