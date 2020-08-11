#pragma once

#include "Function.h"

class TanhFunction : public Function
{
public:
	TanhFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastOutput;

	xt::xarray<double> activationDerivative();
};