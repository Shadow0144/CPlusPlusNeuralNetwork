#pragma once

#include "ErrorFunction.h"

class CrossEntropyFunction : public ErrorFunction
{
public:
	double getError(xt::xarray<double> predicted, xt::xarray<double> actual);
	xt::xarray<double> getDerivativeOfError(xt::xarray<double> predicted, xt::xarray<double> actual);
	// Special case where previous layer is softmax
	xt::xarray<double> getDerivativeOfErrorSoftmax(xt::xarray<double> predicted, xt::xarray<double> actual);
};