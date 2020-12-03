#pragma once

#include "Function.h"

// Maxout
class MaxoutFunction : public Function
{
public:
	MaxoutFunction(size_t incomingUnits, size_t numUnits, size_t numFunctions);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	size_t numFunctions;

	xt::xarray<double> activationDerivative();
};