#pragma once

#include "ActivationFunction.h"

// Rectified Linear Unit - 6
class ReLU6Function : public ActivationFunction
{
public:
	ReLU6Function();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	xt::xarray<double> reLU6(const xt::xarray<double>& z);
};