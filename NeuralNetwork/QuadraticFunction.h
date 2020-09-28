#pragma once

#include "Function.h"

// Quadratic
class QuadraticFunction : public Function
{
public:
	QuadraticFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastZ;

	xt::xarray<double> activationDerivative();
};