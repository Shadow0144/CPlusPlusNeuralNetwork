#pragma once

#include "ActivationFunction.h"

// Swish
class SwishFunction : public ActivationFunction
{
public:
	SwishFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas) const;
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const;

private:
	double activate(double z) const;

	double swish(double z) const;
	xt::xarray<double> swish(const xt::xarray<double>& z) const;

	double sigmoid(double z) const;
	xt::xarray<double> sigmoid(const xt::xarray<double>& z) const;
};