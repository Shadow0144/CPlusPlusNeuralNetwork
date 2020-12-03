#pragma once

#include "Function.h"

class SqueezeFunction : public Function
{
public:
	SqueezeFunction(const std::vector<size_t>& squeezeDims);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<size_t> squeezeDims;
};
