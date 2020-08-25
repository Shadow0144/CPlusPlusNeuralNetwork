#pragma once

#include "Function.h"

// Parametric Rectified Linear Unit
class PReLUFunction : public Function
{
public:
	PReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	double applyBackPropagate();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastOutput;

	xt::xarray<double> activationDerivative();

	xt::xarray<double> PReLU(xt::xarray<double> z);

	double a; // Leak coefficient
	double deltaA;
};