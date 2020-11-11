#pragma once

#include "Function.h"

class FlattenFunction : public Function
{
public:
	FlattenFunction();

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);
};
