#include "AdadeltaOptimizer.h"

const double INTERNAL_BATCH_LIMIT = 20;

using namespace std;

const std::string AdadeltaOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string AdadeltaOptimizer::GAMMA = "gamma"; // Parameter string [OPTIONAL]
const std::string AdadeltaOptimizer::EPSILON = "epsilon"; // Parameter string [OPTIONAL]

AdadeltaOptimizer::AdadeltaOptimizer(vector<NeuralLayer*>* layers, int batchSize, double gamma, double epsilon) : Optimizer(layers)
{
	this->batchSize = batchSize;
	this->gamma = gamma;
	this->epsilon = epsilon;
}

AdadeltaOptimizer::AdadeltaOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(BATCH_SIZE) == additionalParameters.end())
	{
		this->batchSize = -1;
	}
	else
	{
		this->batchSize = additionalParameters[BATCH_SIZE];
	}
	if (additionalParameters.find(GAMMA) == additionalParameters.end())
	{
		this->gamma = 0.9;
	}
	else
	{
		this->gamma = additionalParameters[GAMMA];
	}
	if (additionalParameters.find(EPSILON) == additionalParameters.end())
	{
		this->epsilon = 1e-7;
	}
	else
	{
		this->epsilon = additionalParameters[EPSILON];
	}
}

xt::xarray<double> AdadeltaOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	if (Eg2.find(parameterID) == Eg2.end() || 
		Ew2.find(parameterID) == Ew2.end() ||
		deltaW.find(parameterID) == deltaW.end())
	{
		Eg2[parameterID] = xt::zeros<double>(gradient.shape());
		Ew2[parameterID] = xt::zeros<double>(gradient.shape());
		deltaW[parameterID] = xt::zeros<double>(gradient.shape());
	}
	else { }
	Eg2[parameterID] = (gamma * Eg2[parameterID]) + ((1 - gamma) * xt::pow(gradient, 2.0));
	Ew2[parameterID] = (gamma * Ew2[parameterID]) + ((1 - gamma) * xt::pow(deltaW[parameterID], 2.0));
	auto RMSg = xt::pow(Eg2[parameterID] + epsilon, +0.5);
	auto RMSw = xt::pow(Ew2[parameterID] + epsilon, +0.5);
	deltaW[parameterID] = -(RMSw / RMSg) * gradient;
	return deltaW[parameterID];
}