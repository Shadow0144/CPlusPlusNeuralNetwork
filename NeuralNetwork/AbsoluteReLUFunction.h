#pragma once

#include "Function.h"

// Absolute Rectified Linear Unit / Absolute Value Rectified Linear Unit
class AbsoluteReLUFunction : public Function
{
public:
	AbsoluteReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> absoluteReLU(const xt::xarray<double>& z);
};