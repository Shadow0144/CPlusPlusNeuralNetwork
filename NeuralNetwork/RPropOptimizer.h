#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class RPropOptimizer : public Optimizer
{
public:
	RPropOptimizer(std::vector<NeuralLayer*>* layers, double shrinkAlpha = 0.5, double growAlpha = 1.2, double minAlpha = 0.0001, double maxAlpha = 50.0);
	RPropOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient); // Adjusts the gradient based on the optimizer

	const static std::string MIN_ALPHA; // = "minAlpha"; // Parameter string [OPTIONAL] // Minimum value for any learning rate
	const static std::string MAX_ALPHA; // = "maxAlpha"; // Parameter string [OPTIONAL] // Maximum value for any learning rate
	const static std::string SHRINK_ALPHA; // = "shrinkAlpha"; // Parameter string [OPTIONAL] // Value to multiplicatively decrease a learning rate by
	const static std::string GROW_ALPHA; // = "growAlpha"; // Parameter string [OPTIONAL] // Value to multiplicatively increase a learning rate by

private:
	std::map<long, xt::xarray<double>> alpha; // Learning rates
	double shrinkAlpha; // Value to increase alpha by
	double growAlpha; // Value to decrease alpha by
	double minAlpha; // Minimum value for any alpha
	double maxAlpha; // Maximum value for any alpha

	std::map<long, xt::xarray<double>> g; // Previous gradient
};