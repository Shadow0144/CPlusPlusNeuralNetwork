#pragma once

#include "ActivationFunction.h"

// Softsign
class SoftsignFunction : public ActivationFunction
{
public:
	SoftsignFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> activationDerivative();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	double activate(double z);

	double softsign(double z);
	xt::xarray<double> softsign(const xt::xarray<double>& z);
};