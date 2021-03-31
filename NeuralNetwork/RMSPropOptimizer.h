#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "LossFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class RMSPropOptimizer : public Optimizer
{
public:
	RMSPropOptimizer(std::vector<NeuralLayer*>* layers, int batchSize = -1, double eta = 0.01, double epsilon = 1e-7);
	RMSPropOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient); // Adjusts the gradient based on the optimizer

	virtual inline std::vector<std::string> getHyperparameterStrings()
	{
		return {
			BATCH_SIZE, // = "batchSize"; // Values less than 0 for batch size = N
			ETA, // = "eta"; // Learning rate
			GAMMA, // = "gamma"; // Momentum rate
			EPSILON // = "epsilon"; // Avoids divide-by-zero errors
		};
	}

private:
	double eta; // Learning rate
	double epsilon; // Small term to prevent divide-by-zero errors

	std::map<long, xt::xarray<double>> Eg2; // Decaying average of square gradients
};