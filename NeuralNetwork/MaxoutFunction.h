#pragma once

#include "Function.h"

// Maxout
class MaxoutFunction : public Function
{
public:
	MaxoutFunction(size_t incomingUnits, size_t numUnits, size_t numFunctions);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<size_t> lastIndices;
	xt::xarray<double> lastOutput;
	size_t numFunctions;

	xt::xarray<double> activationDerivative();
};