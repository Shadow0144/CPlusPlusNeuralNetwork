#pragma once

#include "ActivationFunction.h"

class HardSigmoidFunction : public ActivationFunction
{
public:
	HardSigmoidFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> activationDerivative();
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	xt::xarray<double> hard_sigmoid(const xt::xarray<double>& z);
};