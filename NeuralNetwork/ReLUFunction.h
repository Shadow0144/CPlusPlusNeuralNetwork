#pragma once

#include "Function.h"

class ReLUFunction : public Function
{
public:
	ReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastOutput;

	xt::xarray<double> activationDerivative();

	xt::xarray<double> reLU(xt::xarray<double> z);
};