#pragma once

#include "LossFunction.h"

class CrossEntropyLossFunction : public LossFunction
{
public:
	CrossEntropyLossFunction();

	double getLoss(const NeuralNetwork* network, const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;

	virtual void checkForOptimizedGradient(NeuralLayer* finalLayer); // Enable optimized gradient calculation iff the final layer is softmax

private:
	bool useSoftmaxGradient;

	const double EPSILON = pow(10, -8);

	// Standard gradient equation
	xt::xarray<double> getGradientStandard(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;
	// Special optimized case where the final layer is softmax
	xt::xarray<double> getGradientSoftmax(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;
};