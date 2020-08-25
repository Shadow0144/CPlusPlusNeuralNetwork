#pragma once

#include "Function.h"

// Concatenated Rectified Linear Unit
class CReLUFunction : public Function
{
public:
	CReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastOutput;

	xt::xarray<double> activationDerivative();

	xt::xarray<double> CReLU(xt::xarray<double> z);
};