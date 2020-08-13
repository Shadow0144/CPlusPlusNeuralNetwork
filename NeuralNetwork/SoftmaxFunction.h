#pragma once

#include "Function.h"

class SoftmaxFunction : public Function
{
public:
	SoftmaxFunction(std::vector<size_t> numInputs, std::vector<size_t> numOutputs);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	std::vector<int> sumIndices;
	xt::xarray<double> lastOutput;
};