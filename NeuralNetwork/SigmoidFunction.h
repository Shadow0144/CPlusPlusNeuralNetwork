#pragma once

#include "ActivationFunction.h"

// Sigmoid
class SigmoidFunction : public ActivationFunction
{
public:
	SigmoidFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

private:
	double activate(double z) const;

	double sigmoid(double z) const;
	xt::xarray<double> sigmoid(const xt::xarray<double>& z) const;
};