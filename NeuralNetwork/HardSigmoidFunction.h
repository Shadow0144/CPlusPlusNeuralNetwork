#pragma once

#include "Function.h"

class HardSigmoidFunction : public Function
{
public:
	HardSigmoidFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> hard_sigmoid(const xt::xarray<double>& z);
};