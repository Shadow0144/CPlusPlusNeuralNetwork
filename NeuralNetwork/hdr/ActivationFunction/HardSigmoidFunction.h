#pragma once

#include "ActivationFunction/ActivationFunction.h"

class HardSigmoidFunction : public ActivationFunction
{
public:
	HardSigmoidFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

private:
	xt::xarray<double> hard_sigmoid(const xt::xarray<double>& z) const;
};