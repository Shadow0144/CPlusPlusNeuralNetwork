#pragma once

#include "Function.h"

// Swish
class SwishFunction : public Function
{
public:
	SwishFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastOutput;
	xt::xarray<double> lastSigmoid;

	double activate(double z);
	xt::xarray<double> activationDerivative();

	double swish(double z);
	xt::xarray<double> swish(xt::xarray<double> z);

	double sigmoid(double z);
	xt::xarray<double> sigmoid(xt::xarray<double> z);
};