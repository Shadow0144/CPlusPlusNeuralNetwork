#pragma once

#include "Function.h"

class Convolution1DFunction : public Function
{
public:
	Convolution1DFunction(const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride, size_t numKernels);

	xt::xarray<double> feedForward(const xt::xarray<double>& inputs);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<size_t> convolutionShape;
	size_t stride;
	size_t numKernels;
};