#pragma once

#include "Function.h"

// Rectified Linear Unit - n
class ReLUnFunction : public Function
{
public:
	ReLUnFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getN();
	void setN(double n);

private:
	xt::xarray<double> lastOutput;

	double n = 1.0; // Activation limit

	xt::xarray<double> activationDerivative();

	xt::xarray<double> reLUn(xt::xarray<double> z);
};