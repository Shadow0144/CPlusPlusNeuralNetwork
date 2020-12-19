#pragma once

#include "ActivationFunction.h"

// Exponential Linear Unit
class ELUFunction : public ActivationFunction
{
public:
	ELUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> activationDerivative();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

	double getAlpha();
	void setAlpha(double alpha);

private:
	double activate(double z);

	double ELU(double z);
	xt::xarray<double> ELU(const xt::xarray<double>& z);

	double alpha = 0.2;
};