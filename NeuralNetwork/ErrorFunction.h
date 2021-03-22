#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class Optimizer;

class ErrorFunction
{
public:
	virtual double getError(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) = 0;
	virtual xt::xarray<double> getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) = 0;
};