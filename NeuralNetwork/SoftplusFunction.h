#pragma once

#include "Function.h"

// Softplus / Smooth Rectified Linear Unit
class SoftplusFunction : public Function
{
public:
	SoftplusFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getK();
	void setK(double k);

private:
	double activate(double z);
	xt::xarray<double> activationDerivative();

	double softplus(double z);
	xt::xarray<double> softplus(const xt::xarray<double>& z);

	double k = 1.0; // Sharpness coefficient
};