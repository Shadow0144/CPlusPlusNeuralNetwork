#include "RMSPropOptimizer.h"

const double INTERNAL_BATCH_LIMIT = 20;

using namespace std;

const std::string RMSPropOptimizer::ETA = "eta"; // Parameter string [REQUIRED]
const std::string RMSPropOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string RMSPropOptimizer::EPSILON = "epsilon"; // Parameter string [OPTIONAL]

RMSPropOptimizer::RMSPropOptimizer(vector<NeuralLayer*>* layers, double eta, int batchSize, double epsilon) : Optimizer(layers)
{
	this->eta = eta;
	this->batchSize = batchSize;
	this->epsilon = epsilon;
}

RMSPropOptimizer::RMSPropOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(ETA) == additionalParameters.end())
	{
		throw std::invalid_argument(std::string("Missing required parameter: ") +
			"RMSPropOptimizer::ETA" + " (\"" + ETA + "\")");
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

xt::xarray<double> RMSPropOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	if (Eg2.find(parameterID) == Eg2.end())
	{
		Eg2[parameterID] = xt::zeros<double>(gradient.shape());
	}
	else { }
	Eg2[parameterID] = (0.9 * Eg2[parameterID]) + ((0.1) * xt::pow(gradient, 2.0));
	xt::xarray<double> optimizedGradient = -eta * xt::pow(Eg2[parameterID] + epsilon, 0.5) * gradient;
	return optimizedGradient;
}