#include "AdagradOptimizer.h"

using namespace std;

AdagradOptimizer::AdagradOptimizer(vector<NeuralLayer*>* layers, int batchSize, double eta, double epsilon) : Optimizer(layers)
{
	this->batchSize = batchSize;
	this->eta = eta;
	this->epsilon = epsilon;
}

AdagradOptimizer::AdagradOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(BATCH_SIZE) == additionalParameters.end())
	{
		this->batchSize = -1;
	}
	else
	{
		this->batchSize = additionalParameters[BATCH_SIZE];
	}
	if (additionalParameters.find(ETA) == additionalParameters.end())
	{
		this->eta = 0.01;
	}
	else
	{
		this->eta = additionalParameters[ETA];
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

xt::xarray<double> AdagradOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	xt::xarray<double> optimizedGradient;
	if (G.find(parameterID) == G.end())
	{
		G[parameterID] = xt::zeros<double>(gradient.shape());
	}
	else { }
	G[parameterID] += xt::pow(gradient, 2.0);
	optimizedGradient = -eta * (xt::pow((G[parameterID] + epsilon), -0.5)) * gradient;
	return optimizedGradient;
}