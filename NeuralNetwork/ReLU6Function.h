#pragma once

#include "Function.h"

// Rectified Linear Unit - 6
class ReLU6Function : public Function
{
public:
	ReLU6Function(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> reLU6(const xt::xarray<double>& z);
};