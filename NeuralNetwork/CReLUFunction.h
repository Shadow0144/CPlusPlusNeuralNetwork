#pragma once

#include "Function.h"

// Concatenated Rectified Linear Unit
class CReLUFunction : public Function
{
public:
	CReLUFunction(size_t incomingUnits, size_t numUnits);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	xt::xarray<double> activationDerivative();
	std::vector<size_t> getOutputShape();

	xt::xarray<double> CReLU(const xt::xarray<double>& z);
};