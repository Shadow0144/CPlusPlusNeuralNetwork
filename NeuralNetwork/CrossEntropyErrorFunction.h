#pragma once

#include "ErrorFunction.h"

class CrossEntropyErrorFunction : public ErrorFunction
{
public:
	double getError(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);
	xt::xarray<double> getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);
	// Special case where previous layer is softmax
	xt::xarray<double> getDerivativeOfErrorSoftmax(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);
};