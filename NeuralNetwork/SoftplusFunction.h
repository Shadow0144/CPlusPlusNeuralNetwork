#pragma once

#include "Function.h"

// Softplus / SmoothReLU
class SoftplusFunction : public Function
{
public:
	SoftplusFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getK();
	void setK(double k);

private:
	xt::xarray<double> lastOutput;

	xt::xarray<double> activationDerivative();

	xt::xarray<double> softplus(xt::xarray<double> z);

	double k = 1.0; // Sharpness coefficient
};