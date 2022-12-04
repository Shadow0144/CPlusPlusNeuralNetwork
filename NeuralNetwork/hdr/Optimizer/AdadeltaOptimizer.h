#pragma once

#include "Optimizer/Optimizer.h"
#include "NeuralLayer/NeuralLayer.h"
#include "LossFunction/LossFunction.h"
#include "ParameterSet.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class AdadeltaOptimizer : public Optimizer
{
public:
	AdadeltaOptimizer(std::vector<NeuralLayer*>* layers, int batchSize = -1, double gamma = 0.9, double epsilon = 1e-7);
	AdadeltaOptimizer(std::vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	void setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient);

	virtual inline std::vector<std::string> getHyperparameterStrings()
	{
		return {
			BATCH_SIZE, // = "batchSize"; // Values less than 0 for batch size = N
			GAMMA, // = "gamma"; // Momentum rate
			EPSILON // = "epsilon"; // Avoids divide-by-zero errors
		};
	}

private:
	double gamma; // Momentum rate
	double epsilon; // Small term to prevent divide-by-zero errors

	std::map<long, xt::xarray<double>> Eg2; // Decaying average of square gradients
	std::map<long, xt::xarray<double>> Ew2; // Decaying average of square weights
	std::map<long, xt::xarray<double>> deltaW; // The change in weights from the previous timestep
};