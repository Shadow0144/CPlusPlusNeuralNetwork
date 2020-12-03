#pragma once

#include "Function.h"

// Leaky Rectified Linear Unit
class LeakyReLUFunction : public Function
{
public:
	LeakyReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

	double getA();
	void setA(double a);

private:
	xt::xarray<double> activationDerivative();

	xt::xarray<double> leakyReLU(const xt::xarray<double>& z);

	double a = 0.01; // Leak coefficient
};