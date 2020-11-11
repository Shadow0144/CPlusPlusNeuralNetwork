#pragma once

#include "Function.h"

class Convolution1DFunction : public Function
{
public:
	Convolution1DFunction(std::vector<size_t> convolutionShape, size_t inputChannels, size_t stride, size_t numKernels);

	xt::xarray<double> feedForward(xt::xarray<double> inputs);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<size_t> convolutionShape;
	size_t stride;
	size_t numKernels;
	xt::xarray<double> lastOutput;
};