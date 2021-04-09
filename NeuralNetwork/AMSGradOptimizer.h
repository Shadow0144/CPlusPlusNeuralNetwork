#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "LossFunction.h"
#include "ParameterSet.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class AMSGradOptimizer : public Optimizer
{
public:
	AMSGradOptimizer(std::vector<NeuralLayer*>* layers, int batchSize = -1, double eta = 0.01, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
	AMSGradOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	void setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient);

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
	std::map<long, xt::xarray<double>> v; // Estimate of the second movement (uncentered variance)
	std::map<long, xt::xarray<double>> vHat; // Maximum of past squared variance
	long t; // Current iteration
};