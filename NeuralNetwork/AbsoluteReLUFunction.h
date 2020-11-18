#pragma once

#include "Function.h"

// Absolute Rectified Linear Unit / Absolute Value Rectified Linear Unit
class AbsoluteReLUFunction : public Function
{
public:
	AbsoluteReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> absoluteReLU(xt::xarray<double> z);
};