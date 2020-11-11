#pragma once

#include "Function.h"

class FlattenFunction : public Function
{
public:
	FlattenFunction(int numOutputs); // Provide a number of outputs for later layers

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);
};
