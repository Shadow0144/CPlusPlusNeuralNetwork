#pragma once

#include "Optimizer/Optimizer.h"
#include "NeuralLayer/NeuralLayer.h"
#include "LossFunction/LossFunction.h"
#include "ParameterSet.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

// Follow The Regularized Leader (-Proximal) optimizer
class FtrlOptimizer : public Optimizer
{
public:
	FtrlOptimizer(std::vector<NeuralLayer*>* layers, int batchSize = -1, double alpha = 1.0, double beta = 1.0, double lambda1 = 0.001, double lambda2 = 0.001);
	FtrlOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	void setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient);

	virtual inline std::vector<std::string> getHyperparameterStrings()
	{
		return {
			BATCH_SIZE, // = "batchSize"; // Values less than 0 for batch size = N
			ALPHA, // = "alpha"; // Learning rate factor
			BETA, // = "beta"; // Learning rate factor
			LAMDA1, // = "lambda1"; // L1 regularization strength
			LAMDA2, // = "lambda2"; // L2 regularization strength
		};
	}

private:
	double alpha; // Learning rate factor
	double beta; // Learning rate factor
	double lambda1; // L1 regularization strength
	double lambda2; // L2 regularization strength

	std::map<long, xt::xarray<double>> z; // Intermediate
	std::map<long, xt::xarray<double>> n; // Intermediate
};