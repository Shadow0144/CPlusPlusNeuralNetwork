#pragma once

#include "Function.h"

// Rectified Linear Unit
class ReLUFunction : public Function
{
public:
	ReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> reLU(xt::xarray<double> z);
};