#include "AdagradOptimizer.h"

using namespace std;

const std::string AdagradOptimizer::ETA = "eta"; // Parameter string [REQUIRED]
const std::string AdagradOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string AdagradOptimizer::EPSILON = "epsilon"; // Parameter string [OPTIONAL]

AdagradOptimizer::AdagradOptimizer(vector<NeuralLayer*>* layers, double eta, int batchSize, double epsilon) : Optimizer(layers)
{
	this->eta = eta;
	this->batchSize = batchSize;
	this->epsilon = epsilon;
}

AdagradOptimizer::AdagradOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(ETA) == additionalParameters.end())
	{
		throw std::invalid_argument(std::string("Missing required parameter: ") +
			"AdagradOptimizer::ETA" + " (\"" + ETA + "\")");
	}
	else
	{
		this->eta = additionalParameters[ETA];
		if (additionalParameters.find(BATCH_SIZE) == additionalParameters.end())
		{
			this->batchSize = -1;
		}
		else
		{
			this->batchSize = additionalParameters[BATCH_SIZE];
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