#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "LossFunction.h"
#include "ParameterSet.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class RPropOptimizer : public Optimizer
{
public:
	RPropOptimizer(std::vector<NeuralLayer*>* layers, double shrinkAlpha = 0.5, double growAlpha = 1.2, double minAlpha = 0.0001, double maxAlpha = 50.0);
	RPropOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	void setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient);

	virtual inline std::vector<std::string> getHyperparameterStrings()
	{
		return {
			MIN_ALPHA, // = "minAlpha"; // Minimum value for any learning rate
			MAX_ALPHA, // = "maxAlpha"; // Maximum value for any learning rate
			SHRINK_ALPHA, // = "shrinkAlpha"; // Value to multiplicatively decrease a learning rate by
			GROW_ALPHA // = "growAlpha"; // Value to multiplicatively increase a learning rate by
		};
	}

private:
	std::map<long, xt::xarray<double>> alpha; // Learning rates
	double shrinkAlpha; // Value to increase alpha by
	double growAlpha; // Value to decrease alpha by
	double minAlpha; // Minimum value for any alpha
	double maxAlpha; // Maximum value for any alpha

	std::map<long, xt::xarray<double>> prevG; // Previous gradient
};