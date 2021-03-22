#pragma once

#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class Optimizer
{
public:
	Optimizer(std::vector<NeuralLayer*>* layers);
	~Optimizer();

	virtual bool backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets) = 0; // Single step

	void setErrorFunction(ErrorFunction* errorFunction);

protected:
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& inputs); // Updates internal values such as last input and last output

	std::vector<NeuralLayer*>* layers;
	ErrorFunction* errorFunction;
	xt::xarray<double> groundTruth;

	int currentEpoch;

	int maxEpochs;
	double minError;
	double errorConvergenceThreshold;
	double weightConvergenceThreshold;
	bool* stoppingConditionFlags;
};