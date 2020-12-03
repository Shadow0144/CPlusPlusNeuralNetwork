#pragma once

#include "ErrorFunction.h"

class MSEFunction : public ErrorFunction
{
public:
	double getError(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);
	xt::xarray<double> getDerivativeOfError(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);
};