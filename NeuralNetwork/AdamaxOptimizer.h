#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "LossFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class AdamaxOptimizer : public Optimizer
{
public:
	AdamaxOptimizer(std::vector<NeuralLayer*>* layers, int batchSize = -1, double eta = 0.01, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
	AdamaxOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient); // Adjusts the gradient based on the optimizer

	virtual inline std::vector<std::string> getHyperparameterStrings()
	{
		return {
			BATCH_SIZE, // = "batchSize"; // Values less than 0 for batch size = N
			ETA, // = "eta"; // Parameter string // Learning rate
			BETA1, // = "beta1"; // Parameter string // First movement decay rate
			BETA2, // = "beta2"; // Parameter string // Infinite movement decay rate
			EPSILON // = "epsilon"; // Avoids divide-by-zero errors
		};
	}

private:
	double eta; // Learning rate
	double beta1; // First movement decay rate
	double beta2; // Second movement decay rate
	double epsilon; // Small term to prevent divide-by-zero errors

	std::map<long, xt::xarray<double>> m; // Estimate of the first movement (mean)
	std::map<long, xt::xarray<double>> u; // Estimate of the infinite movement
	long t; // Current iteration
};