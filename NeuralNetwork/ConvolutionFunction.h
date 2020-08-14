#pragma once

#include "Function.h"

class ConvolutionFunction : public Function
{
public:
	ConvolutionFunction(std::vector<size_t> numInputs, std::vector<size_t> convolutionShape, int stride);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<size_t> convolutionShape;
	int stride;
};