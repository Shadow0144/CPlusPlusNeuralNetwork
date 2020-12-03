#pragma once

#include "Function.h"

// Gaussian Error Linear Unit
class GELUFunction : public Function
{
public:
	GELUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	double activate(double z);
	xt::xarray<double> activationDerivative();

	double GELU(double z);
	xt::xarray<double> GELU(const xt::xarray<double>& z);
};