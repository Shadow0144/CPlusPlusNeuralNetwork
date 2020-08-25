#pragma once

#include "Function.h"

// Softsign
class SoftsignFunction : public Function
{
public:
	SoftsignFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> lastZ;

	double activate(double z);
	xt::xarray<double> activationDerivative();

	double softsign(double z);
	xt::xarray<double> softsign(xt::xarray<double> z);
};