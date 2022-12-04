#pragma once

#include "LossFunction/LossFunction.h"

class MeanSquareErrorLossFunction : public LossFunction
{
public:
	double getLoss(const NeuralNetwork* network, const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;
	xt::xarray<double> getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const;
};