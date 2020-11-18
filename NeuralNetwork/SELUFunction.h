#pragma once

#include "Function.h"

// Scaled Exponential Linear Unit
class SELUFunction : public Function
{
public:
	SELUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	//xt::xarray<double> mask; // Stores which inputs to the activation function were greather than 0
	double activate(double z);
	xt::xarray<double> activationDerivative();

	double SELU(double z);
	xt::xarray<double> SELU(xt::xarray<double> z);

	const double ALPHA = 1.67326324;
	const double SCALE = 1.05070098;
};