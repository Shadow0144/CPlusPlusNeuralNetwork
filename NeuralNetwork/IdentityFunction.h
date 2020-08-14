#pragma once

#include "Function.h"

class IdentityFunction : public Function
{
public:
	IdentityFunction(size_t numUnits, size_t incomingUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);
};
