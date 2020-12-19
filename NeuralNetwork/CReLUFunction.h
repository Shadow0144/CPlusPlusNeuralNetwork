#pragma once

#include "ActivationFunction.h"

// Concatenated Rectified Linear Unit
class CReLUFunction : public ActivationFunction
{
public: // TODO!!!
	CReLUFunction();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights);

private:
	std::vector<size_t> getOutputShape(std::vector<size_t> outputShape);

	xt::xarray<double> CReLU(const xt::xarray<double>& z);
};