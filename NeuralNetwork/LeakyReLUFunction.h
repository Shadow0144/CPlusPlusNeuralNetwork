#pragma once

#include "Function.h"

// Leaky ReLU / Parametric ReLU
class LeakyReLUFunction : public Function
{
public:
	LeakyReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getA();
	void setA(double a);

private:
	xt::xarray<double> lastOutput;

	xt::xarray<double> activationDerivative();

	xt::xarray<double> leakyReLU(xt::xarray<double> z);

	double a = 0.01; // Leak coefficient
};