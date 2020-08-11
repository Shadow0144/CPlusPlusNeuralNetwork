#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class ErrorFunction
{
public:
	virtual double getError(xt::xarray<double> predicted, xt::xarray<double> actual) = 0;
	virtual xt::xarray<double> getDerivativeOfError(xt::xarray<double> predicted, xt::xarray<double> actual) = 0;
};