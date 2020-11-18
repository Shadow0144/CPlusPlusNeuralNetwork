#pragma once

#include "Function.h"

// Average Pooling 1-D
class AveragePooling1DFunction : public Function
{
public:
	AveragePooling1DFunction(std::vector<size_t> filterShape);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<size_t> filterShape;
};