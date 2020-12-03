#pragma once

#include "Function.h"

// Rectified Linear Unit - n
class ReLUnFunction : public Function
{
public:
	ReLUnFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getN();
	void setN(double n);

private:
	double n = 1.0; // Activation limit

	xt::xarray<double> activationDerivative();

	xt::xarray<double> reLUn(const xt::xarray<double>& z);
};