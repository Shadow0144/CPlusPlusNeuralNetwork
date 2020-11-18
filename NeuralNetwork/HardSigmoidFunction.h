#pragma once

#include "Function.h"

class HardSigmoidFunction : public Function
{
public:
	HardSigmoidFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> hard_sigmoid(xt::xarray<double> z);
};