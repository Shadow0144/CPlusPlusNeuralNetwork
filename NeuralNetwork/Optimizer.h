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

	virtual inline std::vector<std::string> getHyperparameterStrings() = 0;

	// Parameter strings
	const static std::string BATCH_SIZE; // = "batchSize"; // Values less than 0 for batch size = N
	const static std::string ETA; // = "eta"; // Learning rate
	const static std::string ALPHA; // = "alpha"; // Learning rate factor
	const static std::string BETA; // = "beta"; // Learning rate factor
	const static std::string BETA1; // = "beta1"; // First movement decay rate
	const static std::string BETA2; // = "beta2"; // Second movement or infinite movement decay rate
	const static std::string GAMMA; // = "gamma"; // Momentum rate
	const static std::string EPSILON; // = "epsilon"; // Avoids divide-by-zero errors
	const static std::string NESTEROV; // = "nesterov"; // Non-zero values for enabling Nesterov Accelerated Gradient
	const static std::string MIN_ALPHA; // = "minAlpha"; // Minimum value for any learning rate
	const static std::string MAX_ALPHA; // = "maxAlpha"; // Maximum value for any learning rate
	const static std::string SHRINK_ALPHA; // = "shrinkAlpha"; // Value to multiplicatively decrease a learning rate by
	const static std::string GROW_ALPHA; // = "growAlpha"; // Value to multiplicatively increase a learning rate by
	const static std::string LAMDA1; // = "lamda1"; // L1 regularization strength
	const static std::string LAMDA2; // = "lamda2"; // L2 regularization strength

protected:
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& inputs); // Updates internal values such as last input and last output
	virtual void backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets);

	void substituteAllParameters();
	void restoreAllParameters();

	std::vector<NeuralLayer*>* layers;
	ErrorFunction* errorFunction;
	xt::xarray<double> groundTruth;

	int batchSize; // The size of a single batch

	const static double INTERNAL_BATCH_LIMIT;
};