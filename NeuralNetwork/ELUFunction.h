#pragma once

#include "Function.h"

// Exponential Linear Unit
class ELUFunction : public Function
{
public:
	ELUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getAlpha();
	void setAlpha(double alpha);

private:
	double activate(double z);
	xt::xarray<double> activationDerivative();

	double ELU(double z);
	xt::xarray<double> ELU(xt::xarray<double> z);

	double alpha = 0.2;
};