#pragma once

#include "Function.h"

// Average Pooling 3-D
class AveragePooling3DFunction : public Function
{
public:
	AveragePooling3DFunction(std::vector<size_t> filterShape);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<size_t> filterShape;

	xt::xarray<double> lastOutput;
};