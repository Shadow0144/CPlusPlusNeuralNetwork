#pragma once

#include "Function.h"

// Max Pooling 2-D
class MaxPooling2DFunction : public Function
{
public:
	MaxPooling2DFunction(size_t filterSize, size_t stride);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	size_t filterSize;
	size_t stride;

	xt::xarray<double> lastOutput;
};