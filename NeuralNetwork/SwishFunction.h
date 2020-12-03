#pragma once

#include "Function.h"

// Swish
class SwishFunction : public Function
{
public:
	SwishFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	double activate(double z);
	xt::xarray<double> activationDerivative();

	double swish(double z);
	xt::xarray<double> swish(const xt::xarray<double>& z);

	double sigmoid(double z);
	xt::xarray<double> sigmoid(const xt::xarray<double>& z);
};