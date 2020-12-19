#pragma once

#include "ActivationFunction.h"

// Leaky Rectified Linear Unit
class LeakyReLUFunction : public ActivationFunction
{
public:
	LeakyReLUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

	double getA();
	void setA(double a);

private:
	xt::xarray<double> leakyReLU(const xt::xarray<double>& z);

	double a = 0.01; // Leak coefficient
};