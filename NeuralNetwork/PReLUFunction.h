#pragma once

#include "Function.h"

// Parametric Rectified Linear Unit
class PReLUFunction : public Function
{
public:
	PReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> PReLU(const xt::xarray<double>& z);

	double a; // Leak coefficient
	double deltaA;
};