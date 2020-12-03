#pragma once

#include "Function.h"

// Max Pooling 2-D
class MaxPooling2DFunction : public Function
{
public:
	MaxPooling2DFunction(const std::vector<size_t>& filterShape);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<size_t> filterShape;
};