#pragma once

#include "ActivationFunction.h"

// Scaled Exponential Linear Unit
class SELUFunction : public ActivationFunction
{
public:
	SELUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	//xt::xarray<double> mask; // Stores which inputs to the activation function were greather than 0
	double activate(double z);

	double SELU(double z);
	xt::xarray<double> SELU(const xt::xarray<double>& z);

	const double ALPHA = 1.67326324;
	const double SCALE = 1.05070098;
};