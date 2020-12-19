#pragma once

#include "ActivationFunction.h"

// Rectified Linear Unit - n
class ReLUnFunction : public ActivationFunction
{
public:
	ReLUnFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

	double getN();
	void setN(double n);

private:
	double n = 1.0; // Activation limit

	xt::xarray<double> reLUn(const xt::xarray<double>& z);
};