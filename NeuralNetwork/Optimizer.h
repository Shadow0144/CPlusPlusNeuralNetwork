#pragma once

#include "ErrorFunction.h"
#include "ParameterSet.h"

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#pragma warning(pop)

class NeuralLayer;

class Optimizer
{
public:
	Optimizer(std::vector<NeuralLayer*>* layers);
	~Optimizer();

	virtual double backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets); // Single step
	virtual xt::xarray<double> getDeltaWeight(long parameterID, const xt::xarray<double>& gradient) = 0; // Adjusts the gradient based on the optimizer

	virtual void substituteParameters(ParameterSet& parameterSet);
	virtual void restoreParameters(ParameterSet& parameterSet);

	void setErrorFunction(ErrorFunction* errorFunction);

protected:
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& inputs); // Updates internal values such as last input and last output
	virtual double backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets);

	void substituteAllParameters();
	void restoreAllParameters();

	std::vector<NeuralLayer*>* layers;
	ErrorFunction* errorFunction;
	xt::xarray<double> groundTruth;

	int batchSize; // The size of a single batch

	const static double INTERNAL_BATCH_LIMIT;
};