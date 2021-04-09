#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "LossFunction.h"
#include "ParameterSet.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class SGDOptimizer : public Optimizer
{
public:
	SGDOptimizer(std::vector<NeuralLayer*>* layers, int batchSize = -1, double eta = 0.01, double gamma = 0, bool nesterov = false);
	SGDOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	void setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient);

	void substituteParameters(ParameterSet& parameterSet);
	void restoreParameters(ParameterSet& parameterSet);

	inline std::vector<std::string> getHyperparameterStrings() {
		return {
			BATCH_SIZE, // = "batchSize"; // Values less than 0 for batch size = N
			ETA, // = "eta"; // Learning rate
			GAMMA, // = "gamma"; // Momentum rate
			NESTEROV // = "nesterov"; // Non-zero values for enabled
		};
	}

private:
	double eta; // Learning rate
	double gamma; // Momentum rate
	bool nesterov; // Using Nestrov accelerated gradient or not

	std::map<long, xt::xarray<double>> previousVelocity;

	void backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets);
};