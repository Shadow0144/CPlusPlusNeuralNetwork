#pragma once

#include "ActivationFunction.h"

// Sigmoid
class SigmoidFunction : public ActivationFunction
{
public:
	SigmoidFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> activationDerivative();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	double activate(double z);

	double sigmoid(double z);
	xt::xarray<double> sigmoid(const xt::xarray<double>& z);
};