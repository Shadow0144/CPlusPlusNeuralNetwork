#pragma once

#include "Function.h"

// Softmax
class SoftmaxFunction : public Function
{
public:
	SoftmaxFunction(size_t incomingUnits, int axis = -1);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	int axis;
	size_t numOutputs;
	xt::xarray<double> lastOutput;
};