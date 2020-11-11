#pragma once

#include "Function.h"

class SqueezeFunction : public Function
{
public:
	SqueezeFunction(std::vector<size_t> squeezeDims);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<size_t> squeezeDims;
};
