#pragma once

#include "Optimizer.h"
#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

// Follow The Regularized Leader (-Proximal) optimizer
class FtrlOptimizer : public Optimizer
{
public:
	FtrlOptimizer(std::vector<NeuralLayer*>* layers, int batchSize = -1, double alpha = 1.0, double beta = 1.0, double lamda1 = 0.001, double lamda2 = 0.001);
	FtrlOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	double backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets); // Single step
	xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient); // Adjusts the gradient based on the optimizer

	void substituteParameters(ParameterSet& parameterSet);

	virtual inline std::vector<std::string> getHyperparameterStrings()
	{
		return {
			BATCH_SIZE, // = "batchSize"; // Values less than 0 for batch size = N
			ALPHA, // = "alpha"; // Learning rate factor
			BETA, // = "beta"; // Learning rate factor
			LAMDA1, // = "lamda1"; // L1 regularization strength
			LAMDA2, // = "lamda2"; // L2 regularization strength
		};
	}

private:
	double alpha; // Learning rate factor
	double beta; // Learning rate factor
	double lamda1; // L1 regularization strength
	double lamda2; // L2 regularization strength

	std::map<long, xt::xarray<double>> z; // Intermediate
	std::map<long, xt::xarray<double>> n; // Intermediate
	std::map<long, xt::xarray<double>> w; // Current weights
};