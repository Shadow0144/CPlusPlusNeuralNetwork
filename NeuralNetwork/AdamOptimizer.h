#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class AdamOptimizer : public Optimizer
{
public:
	AdamOptimizer(std::vector<NeuralLayer*>* layers, double eta, int batchSize = -1, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
	AdamOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	double backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets); // Single step, returns the sum of the changes in weights
	xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient); // Adjusts the gradient based on the optimizer

	const static std::string ETA; // = "eta"; // Parameter string [REQUIRED] // Learning rate
	const static std::string BATCH_SIZE; // = "batchSize"; // Parameter string [OPTIONAL] // Values less than 0 for batch size = N
	const static std::string BETA1; // = "beta1"; // Parameter string [OPTIONAL] // First movement decay rate
	const static std::string BETA2; // = "beta2"; // Parameter string [OPTIONAL] // Second movement decay rate
	const static std::string EPSILON; // = "epsilon"; // Parameter string [OPTIONAL] // Avoids divide-by-zero errors

private:
	int batchSize; // The size of a single batch
	double eta; // Learning rate
	double beta1; // First movement decay rate
	double beta2; // Second movement decay rate
	double epsilon; // Small term to prevent divide-by-zero errors

	std::map<long, xt::xarray<double>> m; // Estimate of the first movement (mean)
	std::map<long, xt::xarray<double>> v; // Estimate of the second movement (uncentered variance)

	double backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets);
};