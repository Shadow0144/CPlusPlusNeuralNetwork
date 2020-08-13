#pragma once

#include "Function.h"

// Parametric ReLU
class ParametricReLUFunction : public Function
{
public:
	ParametricReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	double applyBackPropagate();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastOutput;

	xt::xarray<double> activationDerivative();

	xt::xarray<double> leakyReLU(xt::xarray<double> z);

	double a; // Leak coefficient
	double deltaA;
};