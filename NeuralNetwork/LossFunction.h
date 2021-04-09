#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class NeuralNetwork;

class LossFunction
{
public:
	virtual double getLoss(const NeuralNetwork* network, const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const = 0;
	virtual xt::xarray<double> getGradient(const xt::xarray<double>& predicted, const xt::xarray<double>& actual) const = 0;

	inline void setL1RegularizationStrength(double lambda1) { this->lambda1 = lambda1; }
	inline double getL1RegularizationStrength() const { return lambda1; }
	inline void setL2RegularizationStrength(double lambda2) { this->lambda2 = lambda2; }
	inline double getL2RegularizationStrength() const { return lambda2; }

protected:
	double lambda1;
	double lambda2;
};