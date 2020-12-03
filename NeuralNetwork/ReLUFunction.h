#pragma once

#include "Function.h"

// Rectified Linear Unit
class ReLUFunction : public Function
{
public:
	ReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> reLU(const xt::xarray<double>& z);
};