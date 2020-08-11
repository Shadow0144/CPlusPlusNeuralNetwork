#pragma once

#include "ErrorFunction.h"

class MSEFunction : public ErrorFunction
{
public:
	double getError(xt::xarray<double> predicted, xt::xarray<double> actual);
	xt::xarray<double> getDerivativeOfError(xt::xarray<double> predicted, xt::xarray<double> actual);
};