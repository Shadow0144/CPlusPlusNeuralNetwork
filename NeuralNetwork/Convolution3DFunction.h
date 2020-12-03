#pragma once

#include "Function.h"

class Convolution3DFunction : public Function
{
public:
	Convolution3DFunction(const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride, size_t numKernels);

	xt::xarray<double> feedForward(const xt::xarray<double>& inputs);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	size_t numKernels;
	std::vector<size_t> convolutionShape;
	size_t stride;
	size_t inputChannels;

	xt::xstrided_slice_vector kernelWindowView;

	xt::xarray<double> convolude(const xt::xarray<double>& f, const xt::xarray<double>& g);
};