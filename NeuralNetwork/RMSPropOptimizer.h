#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class RMSPropOptimizer : public Optimizer
{
public:
	RMSPropOptimizer(std::vector<NeuralLayer*>* layers, double eta, int batchSize = -1, double epsilon = 1e-7);
	RMSPropOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	double backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets); // Single step, returns the sum of the changes in weights
	xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient); // Adjusts the gradient based on the optimizer

	const static std::string ETA; // = "eta"; // Parameter string [REQUIRED] // Learning rate
	const static std::string BATCH_SIZE; // = "batchSize"; // Parameter string [OPTIONAL] // Values less than 0 for batch size = N
	const static std::string GAMMA; // = "gamma"; // Parameter string [OPTIONAL] // Momentum rate
	const static std::string EPSILON; // = "epsilon"; // Parameter string [OPTIONAL] // Avoids divide-by-zero errors

private:
	double eta; // Learning rate
	int batchSize; // The size of a single batch
	double epsilon; // Small term to prevent divide-by-zero errors

	std::map<long, xt::xarray<double>> Eg2; // Decaying average of square gradients

	double backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets);
};