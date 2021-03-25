#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class SGDOptimizer : public Optimizer
{
public:
	SGDOptimizer(std::vector<NeuralLayer*>* layers, double alpha, int batchSize = -1, double momentum = 0, bool nesterov = false);
	SGDOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	double backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets); // Single step, returns the sum of the changes in weights
	xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient); // Adjusts the gradient based on the optimizer

	void substituteParameters(ParameterSet& parameterSet);
	void restoreParameters(ParameterSet& parameterSet);

	const static std::string ALPHA; // = "alpha"; // Parameter string [REQUIRED]
	const static std::string BATCH_SIZE; // = "batchSize"; // Parameter string [OPTIONAL] // Values less than 0 for batch size = N
	const static std::string MOMENTUM; // = "momentum"; // Parameter string [OPTIONAL]
	const static std::string NESTEROV; // = "nesterov"; // Parameter string [OPTIONAL] // Non-zero values for enabled

private:
	double alpha; // Learning rate
	int batchSize; // The size of a single batch
	double momentum; // Gamma
	bool nesterov; // Using Nestrov accelerated gradient or not

	std::map<long, xt::xarray<double>> previousVelocity;

	double backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets);
};