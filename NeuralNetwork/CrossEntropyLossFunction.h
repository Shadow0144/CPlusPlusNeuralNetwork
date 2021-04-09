#pragma once

#include "LossFunction.h"

class CrossEntropyLossFunction : public LossFunction
{
public:
	double getLoss(const NeuralNetwork* network, const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;
	// Special optimized case where the final layer is softmax
	xt::xarray<double> getGradientSoftmax(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;
};