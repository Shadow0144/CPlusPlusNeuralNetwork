#pragma once

#include "LossFunction.h"

class MeanSquareErrorLossFunction : public LossFunction
{
public:
	double getLoss(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);
	xt::xarray<double> getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);
};