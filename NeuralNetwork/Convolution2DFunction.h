#pragma once

#include "Function.h"

class Convolution2DFunction : public Function
{
public:
	Convolution2DFunction(size_t incomingUnits, size_t numFilters, std::vector<size_t> convolutionShape, int stride);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	size_t numFilters;
	std::vector<size_t> convolutionShape;
	int stride;
};