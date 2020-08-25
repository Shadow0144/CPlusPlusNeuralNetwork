#pragma once

#include "Function.h"

// Tanh
class TanhFunction : public Function
{
public:
	TanhFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastOutput;

	double activate(double z);
	xt::xarray<double> activationDerivative();
};